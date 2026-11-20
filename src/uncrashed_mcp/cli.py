"""Command line for the Uncrashed track editor: ./uc [--json] [--pick PATHS] <command> [positionals] [--param value ...]

Every operation of api.OPERATIONS is a command. Arguments are bound from the operation's signature and its
registry hints alone: nothing here knows a particular command, apart from the built-ins (help, last, mcp) and
the stdin / rest / variadic mechanisms the hints select.

Exit codes: 0 ok; 1 the operation raised; 2 usage error, nothing ran; 3 completed but reported failures.
Every non-zero exit writes one stderr line `error: ...`.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import signal
import sys
import types
import typing

from . import __version__, agent, api

PROG = './uc'
EMPTY = inspect.Parameter.empty
BUILTINS = ('help', 'last', 'mcp')
TOPICS = ('guide', 'verbs', 'output')
GLOBAL_FLAGS = ('json', 'pick', 'help', 'version')
REGION = ('x0', 'y0', 'x1', 'y1')
SCALARS = ('str', 'int', 'float', 'bool')
LABELS = {'str': 'str', 'int': 'int', 'float': 'num', 'strs': 'a,b,..', 'floats': 'n,n,..', 'points': 'x,y[,z]',
          'strmap': 'k=v,..', 'map': 'k=v,..', 'maps': 'json'}
LEGEND = ('in the text below name=value means --name value; preview=False means --apply; lists are a,b,c; '
          'JSON in single quotes')
FILE_HINT = f'batch takes a command file; use {PROG} exec for one command'
RECOVER = f"check '{PROG} last' / '{PROG} exec mapinfo' before retrying"
PARTIAL = f"live state may be partial: check '{PROG} exec mapinfo', '{PROG} edit_history', '{PROG} log'"

OPTION = re.compile(r'--[A-Za-z]')
_INT = re.compile(r'[+-]?[0-9]+\Z')
_NUM = re.compile(r'[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
_BOOLS = {'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False}
_FLAG_WORDS = ('0', '1', 'yes', 'no', 'on', 'off')  # read as a flag's value by people, as a positional by the scan
JSON_DEPTH = 32  # no parameter nests deeper; beyond it json and the checks below would hit the recursion limit
_ASSIGNMENT = re.compile(r'-{0,2}([A-Za-z_][A-Za-z0-9_-]*)=(.*)\Z', re.S)

OUTPUT = """
Default output is compact text with the data of --json: nothing is rounded or dropped; only the order of
fields inside rows under an `(every row also has)` line may differ.
  key: value                    one line per field; nested fields are indented two spaces
  key: {a=1 b=two}              small nested object on one line
  key: [a,b,c]                  list of plain values
  assets[8]{id|name|size_m}:    table: 8 rows follow, one per line, cells separated by |
    (every row also has) kind=mesh   fields with the same value in every row, written once instead of
                                as columns (tables of 3 or more rows only)
    NewRow_788|House 02|[9.08,14.5,8.28]
  budgets[14]{*|warn|high}:     table of a keyed object: the * column holds the key
  aabb.min                      dotted column: field `min` of the nested object `aabb`
  - {a=1 b=2}                   list item; key: value lines indented under it are more fields of that same item
  (every row also has) k=v      before - items: fields every item has, left out of each item
  "12" "true" "a|b"             quoted: a string that would otherwise read as a number, keyword or delimiter
exec / batch / log print plain text (the agent's reply), not fields: a failure shows as exit code 3 plus the
stderr `error:` line. A reply without any text prints nothing and the stderr note `(empty reply)`.
--json prints one JSON document instead: the value, {"ok":bool,"text":...} for exec / batch / log, or
{"error":msg} on failure; NaN / Infinity become null (stderr warning). --pick a,b.c keeps only those fields
(c inside b, and inside every list item); a path that matches nothing prints a warning and the full result.
""".strip()


class Usage(Exception):
    """The command line is wrong; nothing was executed."""

    def __init__(self, message: str, op: api.Op | None = None):
        super().__init__(message)
        self.op = op


class _Interrupted(BaseException):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


class _Flags:
    def __init__(self, as_json: bool):
        self.json = as_json
        self.pick: str | None = None
        self.help = False
        self.version = False


# ---------------------------------------------------------------------------
# argument types
# ---------------------------------------------------------------------------


def _kind(annotation) -> str:
    """How a parameter is written on the command line, from its annotation alone."""
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (types.UnionType, typing.Union):
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return _kind(inner[0])
    elif annotation in (str, int, float, bool):
        return annotation.__name__
    elif annotation is dict:
        return 'map'
    elif origin is dict and args == (str, str):
        return 'strmap'
    elif origin is list and len(args) == 1:
        item = args[0]
        if item is str:
            return 'strs'
        if item is float:
            return 'floats'
        if item is dict or typing.get_origin(item) is dict:
            return 'maps'
        if typing.get_origin(item) is list and typing.get_args(item) == (float,):
            return 'points'
    raise TypeError(f'no command-line form for a parameter of type {annotation!r}')


def _number(token: str, what: str) -> float:
    if not _NUM.match(token) or not math.isfinite(float(token)):
        raise Usage(f'{what} expects a finite decimal number, got {token!r}')
    return float(token)


def _scalar(kind: str, token: str, what: str):
    if kind == 'str':
        return token
    if kind == 'float':
        return _number(token, what)
    if kind == 'int':
        if not _INT.match(token):
            raise Usage(f'{what} expects an integer, got {token!r}')
        try:
            return int(token)
        except ValueError:  # more digits than int() converts
            raise Usage(f'{what} expects an integer, got one of {len(token)} characters') from None
    if token.lower() not in _BOOLS:
        raise Usage(f'{what} expects true or false, got {token!r}')
    return _BOOLS[token.lower()]


class _Input:
    """Files and stdin named on the command line. stdin can be read once."""

    def __init__(self):
        self.stdin_used = False

    def read(self, name: str, what: str, hint: str = '') -> str:
        try:
            if name == '-':
                if self.stdin_used:
                    raise Usage(f'{what}: stdin (-) was already read by another argument')
                self.stdin_used = True
                if sys.stdin is None:  # the caller closed fd 0
                    raise Usage(f"{what}: stdin (-) is closed{hint and '; ' + hint}")
                data = sys.stdin.buffer.read() if hasattr(sys.stdin, 'buffer') else sys.stdin.read()
            else:
                with open(name, 'rb') as handle:
                    data = handle.read()
            return data.decode('utf-8-sig') if isinstance(data, bytes) else data
        except OSError as exc:
            raise Usage(f"{what}: cannot read {name!r} ({exc.strerror or exc}){hint and '; ' + hint}") from None
        except UnicodeDecodeError:
            raise Usage(f'{what}: {name!r} is not UTF-8 text') from None


def _checked(found, what: str):
    """A value read as JSON from the command line: finite numbers only, nested no deeper than JSON_DEPTH."""
    pending = [(found, 1)]
    while pending:
        value, depth = pending.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise Usage(f'{what}: NaN and Infinity are not accepted; numbers must be finite')
        if isinstance(value, (dict, list)):
            if depth > JSON_DEPTH:
                raise Usage(f'{what}: JSON nested more than {JSON_DEPTH} levels deep')
            pending += [(item, depth + 1) for item in (value.values() if isinstance(value, dict) else value)]
    return found


def _json(text: str, what: str):
    try:
        found = json.loads(text)
    except RecursionError:
        raise Usage(f'{what}: JSON nested more than {JSON_DEPTH} levels deep') from None
    except ValueError as exc:
        raise Usage(f'{what}: invalid JSON ({exc})') from None
    return _checked(found, what)


def _element(kind: str, item, what: str):
    if kind == 'floats':
        if isinstance(item, str):
            return _number(item.strip(), what)
        if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            return float(item)
        raise Usage(f'{what} expects finite numbers, got {json.dumps(item)}')
    if not isinstance(item, str):
        raise Usage(f'{what} expects strings, got {json.dumps(item)} (quote it inside the JSON list)')
    return item


def _sequence(kind: str, token: str, what: str, source: _Input) -> list:
    """One list token: a,b,c | '[..]' (JSON) | @file | @- (stdin)."""
    if token.startswith('@'):
        text = source.read(token[1:], what).strip()
        found = _json(text, what) if text.startswith('[') else [part for part in re.split(r'[\s,]+', text) if part]
        if not isinstance(found, list) or not found:
            raise Usage(f'{what}: {token} holds no values')
    elif token.lstrip().startswith('['):
        found = _json(token, what)
        if not isinstance(found, list):
            raise Usage(f'{what}: expected a JSON list')
    else:
        found = [part.strip() for part in token.split(',')]
        if '' in found:
            raise Usage(f'{what}: empty value in {token!r} (an intentionally empty list is written [])')
    return [_element(kind, item, what) for item in found]


def _point(parts: list, what: str) -> list[float]:
    if not 2 <= len(parts) <= 3:
        raise Usage(f'{what}: a point is x,y or x,y,z, got {",".join(str(p) for p in parts)!r}')
    return [_element('floats', part, what) for part in parts]


def _points(token: str, what: str, source: _Input) -> list[list[float]]:
    """One point token: x,y[,z] | JSON point or list of points | @file with one point per line or JSON."""
    text = source.read(token[1:], what).strip() if token.startswith('@') else token.strip()
    if text.startswith('['):
        found = _json(text, what)
        if not isinstance(found, list):
            raise Usage(f'{what}: expected a JSON list of points')
        if found and not any(isinstance(item, list) for item in found):
            found = [found]
        if not all(isinstance(item, list) for item in found):
            raise Usage(f'{what}: expected a JSON list of [x,y] / [x,y,z] points')
        points = [_point(item, what) for item in found]
    elif token.startswith('@'):
        points = [_point([part for part in re.split(r'[\s,]+', line) if part], what)
                  for line in text.split('\n') if line.strip()]
    else:
        points = [_point([part.strip() for part in text.split(',')], what)]
    if token.startswith('@') and not points:
        raise Usage(f'{what}: {token} holds no points')
    return points


def _merge(into: dict, new: dict, what: str, trail: str = '') -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(into.get(key), dict):
            _merge(into[key], value, what, f'{trail}{key}.')
        elif key in into:
            raise Usage(f'{what}: {trail}{key} is given twice')
        else:
            into[key] = value
    return into


def _mapping(kind: str, token: str, what: str, source: _Input) -> dict:
    """One dict token: k=v[,k=v] | '{..}' (JSON) | @file (JSON). A bare dict nests dotted keys and reads JSON values."""
    if token.startswith('@') or token.lstrip().startswith('{'):
        found = _json(source.read(token[1:], what) if token.startswith('@') else token, what)
        if not isinstance(found, dict):
            raise Usage(f'{what}: expected a JSON object')
        if token.startswith('@') and not found:
            raise Usage(f'{what}: {token} holds no values')
    else:
        found = {}
        for part in token.split(','):
            key, sep, value = part.partition('=')
            key = key.strip()
            if not sep or not key or (kind == 'map' and '' in key.split('.')):
                raise Usage(f'{what} expects key=value[,key=value], got {part!r} (an intentionally empty one is {{}})')
            if kind == 'map':
                try:
                    value = json.loads(value)
                except RecursionError:
                    raise Usage(f'{what} {key}: JSON nested more than {JSON_DEPTH} levels deep') from None
                except ValueError:
                    pass  # not JSON: the text itself
                else:
                    _checked(value, f'{what} {key}')
                *outer, key = key.split('.')
                for name in reversed(outer):
                    value, key = {key: value}, name
            _merge(found, {key: value}, what)
    if kind == 'strmap' and not all(isinstance(value, str) for value in found.values()):
        raise Usage(f'{what} expects text values, e.g. base=foundation')
    return found


def _records(token: str, what: str, source: _Input) -> list[dict]:
    """One list[dict] token: JSON list of objects, one JSON object, or @file holding either."""
    found = _json(source.read(token[1:], what) if token.startswith('@') else token, what)
    if isinstance(found, dict):
        found = [found]
    if not isinstance(found, list) or not all(isinstance(item, dict) for item in found):
        raise Usage(f'{what} expects a JSON list of objects, e.g. \'[{{"x":0,"y":0}}]\'')
    if token.startswith('@') and not found:
        raise Usage(f'{what}: {token} holds no values')
    return found


def _container(kind: str, token: str, what: str, source: _Input):
    if kind in ('strs', 'floats'):
        return _sequence(kind, token, what, source)
    if kind == 'points':
        return _points(token, what, source)
    if kind == 'maps':
        return _records(token, what, source)
    return _mapping(kind, token, what, source)


# ---------------------------------------------------------------------------
# command line -> keyword arguments
# ---------------------------------------------------------------------------


def _nearest(word: str, candidates) -> str:
    import difflib

    match = difflib.get_close_matches(word, list(candidates), n=1, cutoff=0.6)
    return f'; did you mean {match[0]}?' if match else ''


def _commands() -> dict[str, api.Op]:
    table = dict(api.OPERATIONS)
    for op in api.OPERATIONS.values():
        table.update(dict.fromkeys(op.aliases, op))
    return table


def _option(kinds: dict[str, str], typed: str) -> tuple[str, bool] | None:
    """`--name` as typed -> (parameter, global flag, 'apply' or 'region'; negated), None when unknown."""
    key = typed[2:].replace('-', '_')
    if key in GLOBAL_FLAGS or key in kinds:
        return key, False
    if key == 'apply' and kinds.get('preview') == 'bool':
        return key, False
    if key == 'region' and all(kinds.get(name) == 'float' for name in REGION):
        return key, False
    if key.startswith('no_') and kinds.get(key[3:]) == 'bool':
        return key[3:], True
    return None


def _spellings(kinds: dict[str, str]) -> list[str]:
    known = [f'--{name}' for name in GLOBAL_FLAGS] + [f'--{name}' for name in kinds]
    known += [f'--no-{name}' for name, kind in kinds.items() if kind == 'bool']
    known += [f'--{name}' for name in ('apply', 'region') if _option(kinds, f'--{name}')]
    return known


def _take(tokens: list[str], i: int, typed: str, eq: str, value: str) -> tuple[str, int]:
    """The value of a value-taking option: after `=`, or the next token unless that is an option."""
    if eq:
        return value, i
    if i >= len(tokens):
        raise Usage(f'{typed} needs a value')
    if tokens[i] == '--' or OPTION.match(tokens[i]):
        raise Usage(f'{typed} needs a value but {tokens[i]} follows (a value that starts with -- is written '
                    f'{typed}=VALUE)')
    return tokens[i], i + 1


def _global(flags: _Flags, key: str, typed: str, eq: str, value: str, tokens: list[str], i: int) -> int:
    if key == 'pick':
        if flags.pick is not None:
            raise Usage('--pick given twice')
        flags.pick, i = _take(tokens, i, typed, eq, value)
    elif eq:
        raise Usage(f'{typed} takes no value')
    else:
        setattr(flags, key, True)
    return i


def _scan(op: api.Op, kinds: dict[str, str], tokens: list[str], flags: _Flags):
    """Sort a command's tokens into positionals, option occurrences and the rest remainder; converts nothing.

    positionals: (token, came after an explicit --); options: (target, as typed, raw value or bool).
    """
    positionals: list[tuple[str, bool]] = []
    options: list[tuple[str, str, object]] = []
    remainder: list[str] | None = None
    literal = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        if op.rest and (literal or not (token in ('--', '-h') or OPTION.match(token))):
            remainder = tokens[i - 1:]
            for late in () if literal else remainder:
                if late == '-h' or OPTION.match(late):
                    # also an unknown --word: a mistyped option must not travel to the game inside the command
                    typed = late.partition('=')[0]
                    known = late == '-h' or _option(kinds, typed)
                    unknown = '' if known else f'; {op.name} has no option {typed}{_nearest(typed, _spellings(kinds))}'
                    raise Usage(f'put options before the command: {late} came after {remainder[0]!r} '
                                f'(a command that itself contains {late} goes after --){unknown}')
            break
        if literal:
            positionals.append((token, True))
        elif token == '--':
            literal = True
        elif token == '-h':
            flags.help = True
        elif not OPTION.match(token):
            positionals.append((token, False))
        else:
            typed, eq, value = token.partition('=')
            found = _option(kinds, typed)
            if found is None:
                raise Usage(f'unknown option {typed} for {op.name}{_nearest(typed, _spellings(kinds))}')
            target, negated = found
            follows = tokens[i].lower() if i < len(tokens) else None
            if target in GLOBAL_FLAGS:
                i = _global(flags, target, typed, eq, value, tokens, i)
            elif target == 'apply' or negated:
                name = 'preview' if target == 'apply' else target
                if eq or follows in ('true', 'false'):
                    raise Usage(f'{typed} takes no value; write --{name}=true or --{name}=false')
                if follows in _FLAG_WORDS:
                    raise Usage(f'{typed} takes no value but {tokens[i]!r} follows; write --{name}=true or '
                                f'--{name}=false, or put a positional {tokens[i]!r} before {typed}')
                options.append((name, typed, False))
            elif kinds.get(target) == 'bool':
                if eq:
                    options.append((target, typed, _scalar('bool', value, typed)))
                elif follows in ('true', 'false'):
                    options.append((target, typed, follows == 'true'))
                    i += 1
                elif follows in _FLAG_WORDS:  # would bind the flag true and move the word into a positional
                    raise Usage(f'{typed} {tokens[i]} is ambiguous; write {typed}=true, {typed}=false or '
                                f'--no-{target}, or put a positional {tokens[i]!r} before {typed}')
                else:
                    options.append((target, typed, True))
            else:
                value, i = _take(tokens, i, typed, eq, value)
                options.append((target, typed, value))
    return positionals, options, remainder


def _lines(text: str) -> list[str]:
    """Command-file lines as written; only a final newline is dropped (source line n is entry n-1)."""
    lines = [line.rstrip('\r') for line in text.split('\n')]
    return lines[:-1] if lines and not lines[-1] else lines


def _bind(op: api.Op, params: list[api.Param], kinds: dict[str, str], positionals, options, remainder,
          source: _Input) -> dict:
    """Keyword arguments for the operation; raises Usage for anything that does not bind exactly."""
    values: dict = {}
    origin: dict[str, str] = {}  # parameter -> how it was given first
    numbered: list[int] | None = None

    def claim(name: str, typed: str) -> None:
        first = origin.get(name)
        if first is not None:
            if {first, typed} == {'--apply', '--preview'}:
                raise Usage('--apply and --preview contradict each other; give one')
            raise Usage(f'{typed} given twice' if first == typed else f'{typed} and {first} both set {name}; give one')
        origin[name] = typed

    for name, typed, value in options:
        if name == 'region':
            corners = _sequence('floats', value, typed, source)
            if len(corners) != 4:
                raise Usage(f'{typed} expects x0,y0,x1,y1, got {value!r}')
            for key, corner in zip(REGION, corners):
                claim(key, typed)
                values[key] = corner
        elif kinds[name] in SCALARS:
            claim(name, typed)
            values[name] = value if isinstance(value, bool) else _scalar(kinds[name], value, typed)
        elif name == op.stdin:
            # line-oriented: one command per occurrence, never split on commas or spaces
            if value == '-':
                word = op.aliases[0] if op.aliases else op.name
                raise Usage(f"{typed} - would send the command '-'; to read stdin write {PROG} {word} - or "
                            f'{typed} @-')
            if value.startswith('@'):
                found = _lines(source.read(value[1:], typed, FILE_HINT))
                numbered = list(range(1, len(found) + 1)) if name not in values else None
            else:
                found, numbered = [value], None
            values[name] = values.get(name, []) + found
            origin.setdefault(name, typed)
        else:
            found = _container(kinds[name], value, typed, source)  # repeating a list or dict option appends
            if isinstance(found, dict):
                values[name] = _merge(values.get(name, {}), found, typed)
            else:
                values[name] = values.get(name, []) + found
            origin.setdefault(name, typed)

    for token, literal in () if op.rest else positionals:
        if literal:
            continue
        assignment = _ASSIGNMENT.match(token)
        if assignment:
            key, value = assignment.group(1).replace('-', '_'), assignment.group(2)
            if key in kinds:
                fix = f'--{key}={value}' if kinds[key] == 'bool' else f'--{key} {value}'
                raise Usage(f'{token!r} is not an argument of {op.name}; parameters are options: write {fix}')
            raise Usage(f'{token!r} looks like name=value but {op.name} has no parameter {key!r}'
                        f'{_nearest(key, kinds)}; parameters are written --name value (a value that really '
                        'contains = goes after --)')
        if re.match(r'-[A-Za-z]', token):
            raise Usage(f'{token!r} looks like an option; options are written --name (a value that starts with - '
                        'goes after --)')

    if op.stdin:
        if len(positionals) > 1:
            raise Usage(f'expected one command file, got {len(positionals)} arguments; {FILE_HINT}')
        if positionals and op.stdin in values:
            raise Usage(f'give either a command file or --{op.stdin}, not both')
        if positionals:
            name = positionals[0][0]
            values[op.stdin] = _lines(source.read(name, 'command file', FILE_HINT))
            numbered = list(range(1, len(values[op.stdin]) + 1))
        elif op.stdin not in values:
            raise Usage(f'no command file: give FILE, or - to read stdin; {FILE_HINT}')
        if any('\n' in line or '\r' in line for line in values[op.stdin]):
            raise Usage('a command contains a line break; write one command per line of a command file')
        if not any(line.split() and not line.lstrip().startswith('#') for line in values[op.stdin]):
            raise Usage(f'the command file holds no commands; {FILE_HINT}')
    elif not op.rest:
        slots = list(op.positional)
        for index, (token, _) in enumerate(positionals):
            if op.variadic and index >= slots.index(op.variadic):
                name = op.variadic
            elif index < len(slots):
                name = slots[index]
            else:
                takes = f"at most {len(slots)} ({' '.join(s.upper() for s in slots)})" if slots else 'no'
                surplus = ' '.join(repr(t) for t, _ in positionals[len(slots):])
                raise Usage(f'{op.name} takes {takes} positional arguments, got {len(positionals)}: unexpected '
                            f'{surplus} (quote a value that contains spaces; other parameters are --name value)')
            label = name.upper()
            if name == op.variadic:
                if origin.setdefault(name, label) != label:
                    raise Usage(f'{label} and {origin[name]} both set {name}; give one')
                values[name] = values.get(name, []) + _container(kinds[name], token, label, source)
            else:
                claim(name, label)
                values[name] = _scalar(kinds[name], token, label) if kinds[name] in SCALARS \
                    else _container(kinds[name], token, label, source)

    if op.rest:
        if remainder is not None:
            if op.rest in values:
                raise Usage(f'give the command after the options or as --{op.rest}, not both')
            values[op.rest] = ' '.join(remainder)
        command = values.get(op.rest)
        if command is not None and ('\n' in command or '\r' in command):
            raise Usage(f'one command per {op.aliases[0] if op.aliases else op.name}; use batch')
        if command is not None and not command.split():
            raise Usage('the command is empty')

    missing = [p.name for p in params if p.default is EMPTY and p.name not in values]
    if missing:
        shown = [n.upper() if n in op.positional or n in (op.rest, op.variadic) else 'FILE' if n == op.stdin
                 else f'--{n}' for n in missing]
        raise Usage(f"missing {' '.join(shown)}")
    if numbered is not None and '_lines' in inspect.signature(op.fn).parameters:
        values['_lines'] = numbered
    return values


# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------


def _show(default) -> str:
    if isinstance(default, bool):
        return str(default).lower()
    if isinstance(default, float):
        return repr(default).removesuffix('.0')
    return str(default)


def _usage(op: api.Op) -> str:
    """One line naming every parameter with its type and default."""
    params = op.params()
    kinds = {p.name: _kind(p.annotation) for p in params}
    by_name = {p.name: p for p in params}

    def default(p: api.Param) -> str:
        return '' if p.default in (EMPTY, None) or p.default == '' else f'={_show(p.default)}'

    def bracket(p: api.Param, text: str) -> str:
        return text if p.default is EMPTY else f'[{text}]'

    first = []
    for name in op.positional:
        p, kind = by_name[name], kinds[name]
        if name == op.variadic:
            item = {'strs': 'str', 'floats': 'num'}.get(kind, LABELS[kind])
            first.append(bracket(p, f'{name.upper()}:{item} ...'))
        else:
            first.append(bracket(p, f'{name.upper()}:{LABELS[kind]}{default(p)}'))
    if op.stdin:
        first.append(f'<FILE | - | --{op.stdin} CMD>')
    options = []
    region = all(kinds.get(name) == 'float' for name in REGION)
    for p in params:
        kind = kinds[p.name]
        if p.name in op.positional or p.name in (op.rest, op.stdin) or (region and p.name in REGION[1:]):
            continue
        if region and p.name == REGION[0]:
            options.append('[--region x0,y0,x1,y1]')
        elif kind != 'bool':
            label = LABELS[kind] + ('|json' if kind in ('points', 'map') else '')
            options.append(bracket(p, f'--{p.name} {label}{default(p)}'))
        elif p.name == 'preview' and p.default is True:
            options.append('[--apply]')
        elif p.default is True:
            options.append(f'[--no-{p.name}]')
        elif p.default is False:
            options.append(f'[--{p.name}]')
        else:
            options.append(f'[--{p.name}|--no-{p.name}]')
    last = [f'{op.rest.upper()}...'] if op.rest else []
    return ' '.join([PROG, op.name, *first, *options, *last])


def _brief(op: api.Op) -> str:
    """Positionals as shown in the command list."""
    defaults = {p.name: p.default for p in op.params()}
    parts = []
    for name in op.positional:
        label = f'{name.upper()}...' if name == op.variadic else name.upper()
        parts.append(label if defaults[name] is EMPTY else f'[{label}]')
    if op.stdin:
        parts.append('FILE|-')
    if op.rest:
        parts.append(f'{op.rest.upper()}...')
    return ''.join(f' {part}' for part in parts)


def _previews(op: api.Op) -> bool:
    """A changing operation that only previews unless --apply is given."""
    return op.effect == 'write' and any(p.name == 'preview' and p.default is True for p in op.params())


def _overview() -> str:
    groups: dict[str, list[str]] = {'no': [], 'mixed': [], 'yes': []}
    for op in api.OPERATIONS.values():
        mark = '~' if _previews(op) else '!' if op.effect == 'write' else ' '
        groups[op.live].append(f"{mark} {'|'.join((op.name, *op.aliases))}{_brief(op)}  {op.summary}")
    return '\n'.join([
        f'Uncrashed track editor CLI {__version__}',
        f'usage: {PROG} [--json] [--pick PATHS] <command> [positionals] [--param value ...]',
        'Units: metres and degrees (Z up, yaw 0 = +X, 90 = +Y); exec / batch commands use centimetres.',
        'Arguments: --flag / --no-flag; lists a,b,c or @file; -5 is a value; -- ends options; --name=value for a '
        'value that starts with --.',
        f"Output: compact text ({PROG} help output); --json prints one JSON document; --pick a,b.c keeps fields.",
        'Exit: 0 ok | 1 failed, a live change may be partly applied | 2 usage error, nothing ran | 3 ran but '
        'reported failures. Every failure also prints one stderr line "error: ...".',
        f'More: {PROG} help <command> (parameters, examples) | {PROG} help guide (workflow and cautions: read it '
        f'first) | {PROG} help verbs (agent command language) | {PROG} last (last finished agent reply) | {PROG} mcp',
        '! = changes the scene or files at once: no preview. ~ = changes them only with --apply, previews otherwise.',
        '',
        'No game needed:',
        *groups['no'],
        'A saved map NAME without the game, otherwise the open editor:',
        *groups['mixed'],
        'Game open in the track editor:',
        *groups['yes'],
    ])


def _command_help(op: api.Op) -> str:
    needs = {'no': 'no game', 'mixed': 'no game when given a saved map NAME, otherwise the open track editor',
             'yes': 'the game open in the track editor'}[op.live]
    lines = [f'{op.name}: {op.summary}', f'usage: {_usage(op)}']
    if op.aliases:
        lines.append(f"alias: {', '.join(op.aliases)}")
    effect = ('changes the scene or files only with --apply, previews otherwise' if _previews(op)
              else 'changes the scene or files at once (no preview)' if op.effect == 'write' else 'read-only')
    lines.append(f'needs: {needs}; {effect}')
    if op.caution:
        lines.append(f'caution: {op.caution}')
    if op.examples:
        lines += ['examples:', *(f'  {example}' for example in op.examples)]
    doc = inspect.cleandoc(op.fn.__doc__ or '').replace('server instructions', f'{PROG} help verbs')
    return '\n'.join([*lines, f'({LEGEND})', '', doc])


def _help(topic: str | None) -> str:
    key = (topic or '').replace('-', '_')
    if key in ('', 'help', 'commands'):
        return _overview()
    if key in TOPICS:
        return {'guide': api.GUIDE, 'verbs': api.VERBS, 'output': OUTPUT}[key]
    commands = _commands()
    if key in commands:
        return _command_help(commands[key])
    if key == 'last':
        return (f'{PROG} last: print the last finished agent reply ({agent.OUT}), e.g. after a command timed out. '
                'Its first line starts with the sequence number. While a job still runs it is the reply before '
                'that job, and a stderr note says so.')
    if key == 'mcp':
        return f"{PROG} mcp: run the MCP stdio server (needs the 'mcp' package)"
    raise Usage(f"no help topic {topic!r}{_nearest(key, [*commands, *TOPICS, *BUILTINS])} "
                f"(topics: a command name, {', '.join(TOPICS)})")


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _discard(stream) -> None:
    """Point a stream nothing can be written to at /dev/null, so the interpreter's exit-time flush of what is
    still buffered does not fail too (that would turn every exit code into 120)."""
    try:
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, stream.fileno())
        os.close(null)
    except (OSError, ValueError, AttributeError):
        pass


def _write(text: str) -> BaseException | None:
    """The complete output in one write; returns what kept it from being written, None when it was.

    A closed pipe (`| head`) comes back as BrokenPipeError: the reader has what it wanted, and callers keep
    their exit code and stay silent for it.
    """
    if not text:
        return None
    try:
        sys.stdout.write(text if text.endswith('\n') else text + '\n')
        sys.stdout.flush()
    except (OSError, ValueError, AttributeError) as exc:  # closed pipe, full device, closed or missing stdout
        _discard(sys.stdout)
        return exc
    return None


def _unwritten(failed: BaseException | None) -> str:
    """Why output is missing, for the error line; empty when it was written or the reader simply left."""
    if failed is None or isinstance(failed, BrokenPipeError):
        return ''
    return (failed.strerror or str(failed)) if isinstance(failed, OSError) else 'stdout is closed'


def _show_text(text: str, flags: _Flags) -> int:
    """Print the output of a command that ran nothing (help, version, last)."""
    reason = _unwritten(_write(text))
    return _fail(1, f'the output could not be written ({reason})', flags) if reason else 0


def _note(line: str) -> None:
    """One stderr line. A stderr nobody reads must not change what the command does or its exit code."""
    if sys.stderr is None:  # fd 2 was closed at start; print() would fall back to stdout
        return
    try:
        print(line, file=sys.stderr, flush=True)
    except OSError:  # the reader is gone (`2>&1 | head -1`)
        _discard(sys.stderr)
    except (ValueError, RuntimeError):  # closed stream; re-entered from a signal handler mid-write
        pass


def _dump(value) -> str:
    return json.dumps(value, separators=(',', ':'))


def _dump_result(value) -> tuple[str, int]:
    """(one strict JSON document, how many NaN / Infinity became null). json.dumps alone would write the bare
    words NaN and Infinity, which strict parsers reject; refusing instead would discard a finished result."""
    try:
        return json.dumps(value, separators=(',', ':'), allow_nan=False), 0
    except ValueError:
        pass
    lost = 0

    def finite(item):
        nonlocal lost
        if isinstance(item, float) and not math.isfinite(item):
            lost += 1
            return None
        if isinstance(item, dict):
            return {key: finite(inner) for key, inner in item.items()}
        return [finite(inner) for inner in item] if isinstance(item, list) else item

    return json.dumps(finite(value), separators=(',', ':'), allow_nan=False), lost


def _fail(code: int, message: str, flags: _Flags, op: api.Op | None = None) -> int:
    if flags.json:
        _write(_dump({'error': message}))
    _note(f'error: {message}')
    if op is not None:
        _note(f'usage: {_usage(op)}')
    return code


def _message(exc: BaseException) -> str:
    text = ' '.join(str(exc).split())
    if isinstance(exc, (KeyError, TypeError, IndexError, AttributeError)) or not text:
        text = f'{type(exc).__name__}: {text}' if text else type(exc).__name__
    if isinstance(exc, agent.AgentTimeout):
        if text.endswith('may still run'):
            text += f': {RECOVER}'
        elif 'before retrying' not in text:
            text += f'; the command was sent and may still run: {RECOVER}'
    elif isinstance(exc, agent.AgentError) and not isinstance(exc, agent.AgentNotSent) and exc.reply is None \
            and text.endswith('may have run'):  # the agent went away after the command was written
        text += f': {RECOVER}'
    return text


def _pick_tree(spec: str) -> dict:
    tree: dict = {}
    for path in spec.split(','):
        keys = path.strip().split('.')
        if '' in keys:
            raise Usage(f'--pick expects field paths such as count,assets.id; got {spec!r}')
        node = tree
        for key in keys:
            node = node.setdefault(key, {})
        node[''] = path.strip()
    return tree


_MISS = object()


def _pick(value, node: dict, hits: set):
    """Keep the picked paths of a value; a path applies inside every element of a list on its way."""
    if '' in node:  # the whole value: it covers any narrower path below it
        hits.update(_leaves(node))
        return value
    if isinstance(value, dict):
        kept = {key: _pick(item, node[key], hits) for key, item in value.items() if key in node}
        kept = {key: item for key, item in kept.items() if item is not _MISS}
        return kept or _MISS
    if isinstance(value, list):
        if not value:  # nothing to look into: the path is not wrong
            hits.update(_leaves(node))
            return []
        kept = [_pick(item, node, hits) for item in value]
        return _MISS if all(item is _MISS for item in kept) else [{} if item is _MISS else item for item in kept]
    return _MISS


def _leaves(node: dict) -> list[str]:
    return [path for key, child in node.items() for path in ([child] if key == '' else _leaves(child))]


def _present(op: api.Op, value, flags: _Flags, tree: dict | None) -> tuple[str, list[str], str | None]:
    """(stdout text, stderr notes, reported failure) of a finished operation."""
    notes = []
    if isinstance(value, api.Text):
        failure = None if value.ok else value.failure or 'the agent reported a failure'
        if tree:
            notes.append('warning: --pick does not apply to a text reply; printing all of it')
        if not value.text and not flags.json:  # nothing on stdout: tell it apart from a command that failed
            notes.append('(empty reply)')
        return (_dump({'ok': value.ok, 'text': value.text}) if flags.json else value.text), notes, failure
    failure = None
    if op.check is not None:
        try:
            failure = op.check(value)
        except Exception as exc:
            failure = f'could not check the reply for failures ({_message(exc)})'
    if isinstance(value, dict) and value.get('preview') is True:
        notes.append('preview only: nothing changed (re-run with --apply)')
    if tree:
        hits: set = set()
        picked = _pick(value, tree, hits)
        missed = [path for path in _leaves(tree) if path not in hits]
        if missed or picked is _MISS:
            keys = ', '.join(value) if isinstance(value, dict) else ''
            notes.append(f"warning: --pick {','.join(missed)} matched nothing (top-level keys: {keys}); "
                         'printing the full result')
        else:
            value = picked
    if flags.json:
        text, lost = _dump_result(value)
        if lost:
            notes.append(f'warning: {lost} non-finite numbers (NaN / Infinity) are null in --json; the text output '
                         'keeps them')
        return text, notes, failure
    from .render import render

    return render(value), notes, failure


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


class _Signals:
    """SIGINT / SIGTERM / SIGHUP raise in the main thread so `finally` blocks run.

    A changing operation gets its first signal deferred (it finishes and prints its result), because an
    exception thrown into the middle of an apply skips the rollbacks that only catch Exception. That holds
    only once this process has sent a command: before that (validation, the wait for the agent lock or for
    an earlier job) nothing can be half applied, so the signal ends the command at once and nothing is sent.
    """

    def __init__(self, name: str, defer: bool):
        self.name = name
        self.defer = defer
        self.pending: int | None = None
        self.finished = False
        self.waiting = False  # set by the on_lock_wait hook: the operation is in a wait that comes before sending
        self._sent = agent.sent_count
        self._saved: dict = {}

    def __enter__(self):
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                if signal.getsignal(signum) is not signal.SIG_IGN:  # nohup / background: stay ignored
                    self._saved[signum] = signal.signal(signum, self._handle)
            except (ValueError, OSError):  # not the main thread
                pass
        return self

    def __exit__(self, *exc):
        for signum, handler in self._saved.items():
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)

    def sent(self) -> bool:
        """Whether this operation has written a command to the agent (agent.py counts just before the write)."""
        return agent.sent_count != self._sent

    def _handle(self, signum, frame):
        if self.finished:  # the result is being printed: the first signal waits for that, a second one ends it
            if self.pending is not None:
                raise _Interrupted(signum)
            self.pending = signum
        elif self.defer and self.pending is None and self.sent():
            _note(f'{signal.Signals(signum).name} received: finishing {self.name} first, then exiting '
                  '(a second signal aborts it and may leave the change partly applied)')
            self.pending = signum
        else:
            raise _Interrupted(signum)


def _digest(op: api.Op, kwargs: dict) -> str:
    """Identity of what a command will send: the operation and its bound arguments, after files and stdin were
    read. The command line cannot serve: `batch -` and `batch plan.txt` are the same words for different work."""
    return hashlib.sha256((op.name + json.dumps(kwargs, sort_keys=True, default=str)).encode()).hexdigest()


def _wait_hook(digest: str | None, signals: _Signals):
    """agent.on_lock_wait while an operation runs: it is called on every poll of a wait that precedes sending."""
    def waiting(holder: dict | None, waited: float) -> None:
        signals.waiting = True
        # Holders without a digest (the MCP server, older processes, previews) never match.
        if digest and holder and holder['pid'] != os.getpid() and holder.get('digest') == digest:
            raise agent.AgentNotSent(f"the same command is already running as pid {holder['pid']}; "
                                     'nothing was sent: wait for it instead of re-sending')
    return waiting


def _report(op: api.Op, value, flags: _Flags, tree: dict | None) -> tuple[int, bool]:
    """Print a finished operation's result -> (exit code, whether stdout took all of it)."""
    try:
        text, notes, failure = _present(op, value, flags, tree)
    except Exception as exc:
        return _fail(1, f'{op.name} finished but its result could not be printed ({_message(exc)})', flags), False
    failed = _write(text)
    for note in notes:
        _note(note)
    reason = _unwritten(failed)
    if reason:  # not the operation's failure: say that it ran, or the caller repeats it
        _note(f'error: {op.name} finished but its output could not be written ({reason})'
              + (f'; it reported: {failure}' if failure else ''))
    elif failure:
        _note(f'error: {failure}')
    return (3 if failure else 1 if reason else 0), failed is None


def _run(op: api.Op, kwargs: dict, flags: _Flags, tree: dict | None) -> int:
    signals = _Signals(op.name, defer=op.effect == 'write')
    # The guard is for commands that change something at once; a preview may run twice.
    acts = op.effect == 'write' and not (_previews(op) and kwargs.get('preview', True))
    digest = _digest(op, kwargs) if acts else None
    hook, extra = agent.on_lock_wait, agent.holder_extra
    agent.on_lock_wait = _wait_hook(digest, signals)
    if digest:
        agent.holder_extra = {**extra, 'digest': digest}
    try:
        with signals:
            try:
                value = api.call(op.name, **kwargs)
            except Exception as exc:
                if os.environ.get('UNCRASHED_DEBUG', '0') not in ('', '0'):
                    import traceback

                    traceback.print_exc()
                signals.finished = True
                code, printed = _fail(1, _message(exc), flags), True
            else:
                signals.finished = True  # from here on a signal waits until the result is printed
                code, printed = _report(op, value, flags, tree)
            if signals.pending is None:
                return code
            if code == 0:
                _note(f'error: {signal.Signals(signals.pending).name} received during {op.name}: the operation '
                      + ('finished and its result is printed' if printed else 'finished; its output was not read')
                      + ('; do not re-run it' if signals.defer else ''))
            return 128 + signals.pending
    except _Interrupted as stop:
        if signals.finished:  # a second signal while the result was being printed
            _discard(sys.stdout)  # what is still buffered must not block the exit
            _note(f'error: {signal.Signals(stop.signum).name} received while printing the result of {op.name}: the '
                  'operation finished, its output is incomplete' + ('; do not re-run it' if signals.defer else ''))
            return 128 + (signals.pending or stop.signum)
        if signals.sent():
            message = f'interrupted during {op.name}; {PARTIAL}'
        elif signals.waiting:
            message = 'interrupted while waiting; nothing was sent'
        else:  # no command was written, so nothing live can be half done
            message = f'interrupted during {op.name}' + ('; nothing was sent' if op.live != 'no' else '')
        return _fail(128 + stop.signum, message, flags)
    finally:
        agent.on_lock_wait, agent.holder_extra = hook, extra


def _job_note() -> None:
    """`last` shows the reply file, which the agent replaces only when a job ends."""
    pid = agent.game_pid()
    if agent._job_running(agent._game_started(pid) if pid is not None else None):
        _note('note: an agent job is still running; this is the reply of the job before it (the first line '
              'starts with its sequence number)')


def _builtin(name: str, rest: list[str], flags: _Flags) -> int:
    words = []
    i = 0
    while i < len(rest):
        token = rest[i]
        i += 1
        typed, eq, value = token.partition('=')
        if token == '-h':
            flags.help = True
        elif OPTION.match(token):
            if typed[2:] not in GLOBAL_FLAGS:
                raise Usage(f'unknown option {typed} for {name}')
            i = _global(flags, typed[2:], typed, eq, value, rest, i)
        else:
            words.append(token)
    if len(words) > (name == 'help'):
        raise Usage(f"{name} takes {'one topic' if name == 'help' else 'no arguments'}, got {' '.join(words)!r}")
    if flags.version:
        return _show_text(f'uncrashed {__version__}', flags)
    if name == 'help' or flags.help:
        return _show_text(_help(words[0] if words else None if name == 'help' else name), flags)
    if name == 'last':
        _job_note()
        try:
            text = agent.OUT.read_text(errors='replace')
        except OSError as exc:
            return _fail(1, f'no previous agent reply: cannot read {agent.OUT} ({exc.strerror or exc})', flags)
        return _show_text(_dump({'ok': True, 'text': text}) if flags.json else text, flags)
    try:
        from .server import main as serve
    except ModuleNotFoundError as exc:
        if exc.name != 'mcp' and not (exc.name or '').startswith('mcp.'):
            raise
        return _fail(1, "the MCP server needs the 'mcp' package: pip install 'uncrashed-mcp[mcp]'", flags)
    serve()
    return 0


def _dispatch(argv: list[str], flags: _Flags) -> int:
    command = None
    i = 0
    while i < len(argv) and command is None:
        token = argv[i]
        i += 1
        typed, eq, value = token.partition('=')
        if token == '-h':
            flags.help = True
        elif OPTION.match(token):
            if typed[2:] not in GLOBAL_FLAGS:
                raise Usage(f'unknown option {typed} before the command; only --json, --pick PATHS, --help and '
                            '--version go there')
            i = _global(flags, typed[2:], typed, eq, value, argv, i)
        elif token == '--':
            raise Usage('-- before the command; the command comes first')
        else:
            command = token
    rest = argv[i:]
    if command is None:
        return _show_text(f'uncrashed {__version__}' if flags.version else _overview(), flags)
    key = command.replace('-', '_')
    if key in BUILTINS:
        return _builtin(key, rest, flags)
    commands = _commands()
    if key not in commands:
        raise Usage(f"unknown command {command!r}{_nearest(key, [*commands, *BUILTINS])} ('{PROG} help' lists "
                    'the commands)')
    op = commands[key]
    try:
        params = op.params()
        kinds = {p.name: _kind(p.annotation) for p in params}
        positionals, options, remainder = _scan(op, kinds, rest, flags)
        if flags.version:
            return _show_text(f'uncrashed {__version__}', flags)
        if flags.help:
            return _show_text(_command_help(op), flags)
        tree = _pick_tree(flags.pick) if flags.pick is not None else None
        kwargs = _bind(op, params, kinds, positionals, options, remainder, _Input())
    except Usage as exc:
        exc.op = op
        raise
    return _run(op, kwargs, flags, tree)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
        except (AttributeError, ValueError, OSError):
            pass
    before_rest = argv[:argv.index('--')] if '--' in argv else argv
    asked = any(token == '--json' or token.startswith('--json=') for token in before_rest)  # also a wrong --json=x
    flags = _Flags(os.environ.get('UNCRASHED_FORMAT', '').lower() == 'json' or asked)
    try:
        return _dispatch(argv, flags)
    except Usage as exc:
        return _fail(2, ' '.join(str(exc).split()), flags, exc.op)  # one line, whatever a token contained
    except KeyboardInterrupt:
        return _fail(130, 'interrupted', flags)


if __name__ == '__main__':
    raise SystemExit(main())
