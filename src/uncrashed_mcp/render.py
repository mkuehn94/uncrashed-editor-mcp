"""Compact, lossless text form of JSON values (the CLI's default output) and its decoder.

`render(value)` writes any JSON value (dict / list / str / int / float / bool / None, string keys);
`parse(text)` reads it back, and `parse(render(v)) == v` holds exactly, including number types
(`1` / `1.0` / `-0.0`) and booleans. Nothing is rounded or dropped. Key order is kept everywhere
except in rows under an `(every row also has)` line: there the hoisted keys come back last.

Grammar. Lines are joined by a newline; nesting is two spaces per level; there are no blank lines.

Scalars
  `true` `false` `null`; numbers exactly as `json.dumps` writes them; strings bare or JSON-quoted.
  A string is quoted in every position when it is empty; has leading or trailing whitespace;
  contains a control character (newline, tab, U+0000-1F, U+007F-9F, U+2028, U+2029) or a lone
  surrogate; looks like a number (also loose forms such as `007`, `+5`, `.5`, `1_000`) or equals
  `true`, `false`, `null`, `NaN`, `Infinity`, `-Infinity`; starts with `[`, `{`, `"`, `#`, `- ` or
  `(every row also has)`. A bare token is therefore a number only if it is a JSON number, and a
  string otherwise. Each position adds its own delimiters:
    value of a `key: value` line    none: it runs to the end of the line
    key of a `key: value` line      a space, `[`, or a trailing `:`
    top level and `- ` item         a first word ending in `:` (reads as `key: value` / `key:`),
                                    or the shape of a table header `word[N]{...}:`
    inline list element             `,` `[` `]` `"`
    inline dict key or value        space `=` `{` `}` `"`
    table cell                      `|` `"`
    table column name               `|` `{` `}` `"`, and the name `*`
  An inline list inside a table row or an inline dict also quotes for the enclosing delimiters,
  so a row splits on every `|` outside quotes and an inline dict on every space outside quotes.

Containers
  `{}` and `[]` are the empty containers. A "short" value is a scalar or an inline list.
  inline list   `[a,b,c]`, `[[1,2],[3,4]]`: a list of scalars and/or lists of scalars.
                Exception: a list of strings with an item longer than 30 characters is written
                as one `- item` line per element and is not a short value.
  inline dict   `{a=1 b=two c="x y" d=[1,2]}`: a nested dict with 1..12 keys whose values are
                all short. The top-level dict is never inline.
  dict          one `key: value` line per entry. `value` is a short value or an inline dict;
                any other value follows on indented lines under `key:` (or as a table).
  list          one `- ` item per element. A short value stays on the dash line; a list item
                starts on the dash line (`- - x`); a dict item is one of
                    - {a=1 b=2}        all values short (any number of keys)
                    - {a=1 b=2}        at least 2 leading short values: those on the dash line,
                      c:               the remaining entries as dict lines aligned under the `{`,
                        ...            two columns right of the dash like every field of an item
                    - c:               otherwise: first entry on the dash line, the others
                        ...            aligned under it
                      d: 4
  table         a list of at least 2 dicts with identical keys in identical order, no key
                containing `.`, each value a scalar, a list of scalars and/or lists of scalars
                (always inline here), or a non-empty dict of those whose keys are identical in
                every row (flattened into dotted columns such as `aabb.min`):
                    assets[8]{id|name|size_m}:
                      (every row also has) kind=mesh material=null
                      NewRow_788|House 02|[9.08,14.5,8.28]
                `[8]` is the row count. A table needs at most 10 columns and cells of at most 80
                characters. With at least 3 rows, the columns whose value is the same in every
                row (never all of them) can be hoisted: they leave the header and the rows and
                are written once, like the inside of an inline dict, on the `(every row also
                has)` line; each row has those fields too, after its own columns. They are
                hoisted when the table fits only without them, or fits either way and is
                shorter hoisted. Without a fit the list form is used; with at least 3 rows its
                first line is an `(every row also has)` line holding the constant top-level
                keys, left out of every item, if that is shorter.
  keyed table   a nested dict with at least 2 entries whose values qualify as table rows:
                `budgets[14]{*|warn|high}:`, the dict key in the `*` column, same hoisting and
                limits (the key column counts). Without a fit it is an ordinary dict.
  At the top level and in a `- ` item a table header has no key: `[8]{id|name}:`.
"""
from __future__ import annotations

import json
import math
import re

MAX_COLUMNS = 10
MAX_CELL = 80
MAX_INLINE_KEYS = 12
LONG_ITEM = 30
HOIST_ROWS = 3
HOIST = '(every row also has)'

_RAW = r'\x7f-\x9f\U00002028\U00002029\U0000d800-\U0000dfff'  # json.dumps leaves these unescaped
_BASE = (r'\A\Z|\A\s|\s\Z|[\x00-\x1f' + _RAW + r']|\A[\[{"#]|\A- |\A' + re.escape(HOIST) +
         r'|\A(?:true|false|null|NaN|-?Infinity)\Z'
         r'|\A[-+]?(?:\d[\d_]*\.?[\d_]*|\.\d+)(?:[eE][-+]?\d+)?\Z')
_ENTRY = r'[^ ]*:(?: |\Z)|[^ \[]*\[[0-9]+\]\{.*\}:\Z'
_VALUE = re.compile(_BASE)
_ITEM = re.compile(_BASE + r'|\A(?:' + _ENTRY + ')')
_KEY = re.compile(_BASE + r'|[ \[]|:\Z')
_LIST = re.compile(_BASE + r'|[,\[\]"]')
_CELL = re.compile(_BASE + r'|[|"]')
_CELL_LIST = re.compile(_BASE + r'|[,\[\]|"]')
_PAIR = re.compile(_BASE + r'|[ ={}"]')
_PAIR_LIST = re.compile(_BASE + r'|[,\[\] ={}"]')
_COLUMN = re.compile(_BASE + r'|[|{}"]|\A\*\Z')
_ESCAPE = re.compile('[' + _RAW + ']')


def render(value) -> str:
    out: list[str] = []
    _value('', value, 0, out, False, True)
    return '\n'.join(out)


def _string(text: str, needs_quotes: re.Pattern) -> str:
    if not needs_quotes.search(text):
        return text
    quoted = json.dumps(text, ensure_ascii=False)
    return _ESCAPE.sub(lambda m: f'\\u{ord(m.group()):04x}', quoted)


def _scalar(value, needs_quotes: re.Pattern) -> str:
    if isinstance(value, str):
        return _string(value, needs_quotes)
    if value is None:
        return 'null'
    if value is True:
        return 'true'
    if value is False:
        return 'false'
    if isinstance(value, int):
        return int.__repr__(value)
    if isinstance(value, float):
        if math.isnan(value):
            return 'NaN'
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return float.__repr__(value)
    raise TypeError(f'not a JSON value: {type(value).__name__}')


def _key(key) -> str:
    if not isinstance(key, str):
        raise TypeError(f'dict keys must be str, not {type(key).__name__}')
    return key


def _inline_list(value: list) -> bool:
    for item in value:
        if isinstance(item, list):
            for inner in item:
                if isinstance(inner, (list, dict)):
                    return False
        elif isinstance(item, dict):
            return False
    return True


def _long_strings(value: list) -> bool:
    return any(isinstance(v, str) and len(v) > LONG_ITEM for v in value) and all(isinstance(v, str) for v in value)


def _cell_value(value) -> bool:
    return _inline_list(value) if isinstance(value, list) else not isinstance(value, dict)


def _short(value) -> bool:
    if isinstance(value, list):
        return _inline_list(value) and not _long_strings(value)
    return not isinstance(value, dict)


def _list(value: list, needs_quotes: re.Pattern) -> str:
    return '[' + ','.join(_list(v, needs_quotes) if isinstance(v, list) else _scalar(v, needs_quotes)
                          for v in value) + ']'


def _cell(value) -> str:
    return _list(value, _CELL_LIST) if isinstance(value, list) else _scalar(value, _CELL)


def _pairs(items) -> str:
    return ' '.join(_string(_key(k), _PAIR) + '=' + (_list(v, _PAIR_LIST) if isinstance(v, list) else _scalar(v, _PAIR))
                    for k, v in items)


def _size(lines: list[str]) -> int:
    return sum(map(len, lines)) + len(lines)


def _value(head: str, value, indent: int, out: list[str], keyed: bool, top: bool = False) -> None:
    """Append `value`; `head` is the text already on its first line, whose content sits at `indent`."""
    sep = ': ' if keyed else ''
    if isinstance(value, dict):
        if not value:
            out.append(head + sep + '{}')
            return
        items = list(value.items())
        if not top:
            short = next((i for i, (_, v) in enumerate(items) if not _short(v)), len(items))
            if short == len(items) and (not keyed or short <= MAX_INLINE_KEYS):
                out.append(f'{head}{sep}{{{_pairs(items)}}}')
                return
            if len(items) >= 2 and _table(head, list(value.values()), indent, out, keyed, list(value)):
                return
            if not keyed and short >= 2:
                out.append(f'{head}{{{_pairs(items[:short])}}}')
                head = '  ' * indent
                items = items[short:]
        if keyed:
            out.append(head + ':')
            indent += 1
            head = '  ' * indent
        for key, item in items:
            _value(head + _string(_key(key), _KEY), item, indent, out, True)
            head = '  ' * indent
    elif isinstance(value, list):
        if not value:
            out.append(head + sep + '[]')
            return
        if _inline_list(value):
            if not _long_strings(value):
                out.append(head + sep + _list(value, _LIST))
                return
        elif len(value) >= 2 and _table(head, value, indent, out, keyed, None):
            return
        if keyed:
            out.append(head + ':')
            indent += 1
            head = '  ' * indent
        for item in value:
            _value(head + '- ', item, indent + 1, out, False)
            head = '  ' * indent
    else:
        out.append(head + sep + _scalar(value, _VALUE if keyed else _ITEM))


def _table(head: str, rows: list, indent: int, out: list[str], keyed: bool, names: list | None) -> bool:
    """Write `rows` as a table (keyed by `names` for a dict of rows) or as hoisted items; False if neither is used."""
    first = rows[0]
    if not isinstance(first, dict) or not first:
        return False
    keys = list(first)
    for row in rows:
        if not isinstance(row, dict) or list(row) != keys:
            return False
    columns: list[tuple[str, list, int]] = []  # name, value per row, index of the row key or -1 when flattened
    for index, key in enumerate(keys):
        if '.' in _key(key):
            return False
        value = first[key]
        if isinstance(value, dict):
            inner = list(value)
            if not inner:
                return False
            for row in rows:
                if not isinstance(row[key], dict) or list(row[key]) != inner:
                    return False
            for sub in inner:
                if '.' in _key(sub):
                    return False
                columns.append((f'{key}.{sub}', [row[key][sub] for row in rows], -1))
        else:
            columns.append((key, [row[key] for row in rows], index))
    for _, values, _ in columns:
        for value in values:
            if not _cell_value(value):
                return False
    count = len(rows)
    texts = [[_cell(v) for v in values] for _, values, _ in columns]
    constant = [count >= HOIST_ROWS and cells.count(cells[0]) == count for cells in texts]
    if all(constant):
        constant = [False] * len(columns)
    hoisted = [i for i, c in enumerate(constant) if c]
    shown = [i for i, c in enumerate(constant) if not c]
    lead = None if names is None else [_string(_key(k), _CELL) for k in names]
    long = [any(len(c) > MAX_CELL for c in cells) for cells in texts]
    room = MAX_COLUMNS - (lead is not None)
    if (len(shown) <= room and not any(long[i] for i in shown)
            and (lead is None or all(len(c) <= MAX_CELL for c in lead))):
        titles = [_string(name, _COLUMN) for name, _, _ in columns]
        pad = '  ' * (indent + 1)
        line = ''
        if hoisted:
            line = f'{pad}{HOIST} {_pairs((columns[i][0], columns[i][1][0]) for i in hoisted)}'
            saved = sum(len(titles[i]) + 1 + count * (len(texts[i][0]) + 1) for i in hoisted)
            if len(columns) <= room and not any(long) and saved <= len(line) + 1:
                shown, line = list(range(len(columns))), ''  # fits without hoisting and is no longer
        header = ['*'] if lead is not None else []
        header += [titles[i] for i in shown]
        out.append(f'{head}[{count}]{{{"|".join(header)}}}:')
        if line:
            out.append(line)
        for r in range(count):
            cells = [texts[i][r] for i in shown]
            out.append(pad + '|'.join(cells if lead is None else [lead[r]] + cells))
        return True
    top = [index for (_, _, index), c in zip(columns, constant) if c and index >= 0]
    if lead is not None or not top:
        return False
    if keyed:
        indent += 1
    pad = '  ' * indent
    dropped = {keys[i] for i in top}
    pairs = _pairs((keys[i], first[keys[i]]) for i in top)
    short = [f'{pad if keyed else head}{HOIST} {pairs}']
    saved = -_size(short)
    inline = all(_short(first[keys[i]]) for i in top)
    for row in rows:
        rest = {k: v for k, v in row.items() if k not in dropped}
        start = len(short)
        _value(pad + '- ', rest, indent + 1, short, False)
        if inline and all(map(_short, rest.values())):
            saved += len(pairs) + 1  # one `- {...}` line with or without the hoisted pairs
        else:
            whole: list[str] = []
            _value(pad + '- ', row, indent + 1, whole, False)
            saved += _size(whole) - _size(short[start:])
    if saved <= 0:
        return False
    if keyed:
        out.append(head + ':')
    out += short
    return True


_QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"')
_NUMBER = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?')
_HEADER = re.compile(r'\[([0-9]+)\]\{(.*)\}:')
_HOISTED = re.compile(re.escape(HOIST) + r' (.+)')
_ENTRY_LINE = re.compile(_ENTRY)
_WORDS = {'true': True, 'false': False, 'null': None,
          'NaN': float('nan'), 'Infinity': float('inf'), '-Infinity': float('-inf')}


def parse(text: str, reordered: list | None = None):
    """Reference decoder for `render`; raises ValueError on malformed text.

    `reordered`, when given, receives the dicts whose key order may differ from the rendered value:
    the rows read under an `(every row also has)` line and, in a table, their dotted-column dicts.
    """
    if text.endswith('\n'):
        text = text[:-1]
    lines = []
    for raw in text.split('\n'):
        body = raw.lstrip(' ')
        lines.append((len(raw) - len(body), body))
    reader = _Reader(lines, [] if reordered is None else reordered)
    value = reader.block(0)
    if reader.at < len(lines):
        reader.fail('unexpected line')
    return value


def _bare(token: str):
    if token in _WORDS:
        return _WORDS[token]
    if _NUMBER.fullmatch(token):
        return float(token) if any(c in token for c in '.eE') else int(token)
    if _VALUE.search(token):
        raise ValueError(f'bad bare string {token!r}')
    return token


def _quoted(text: str, at: int = 0) -> tuple[str, int]:
    match = _QUOTED.match(text, at)
    if not match:
        raise ValueError(f'bad quoted string in {text!r}')
    return json.loads(match.group()), match.end()


def _token(text: str) -> str:
    """A column name or table key: always a string."""
    if text.startswith('"'):
        value, end = _quoted(text)
        if end != len(text):
            raise ValueError(f'bad quoted string {text!r}')
        return value
    if not text:
        raise ValueError('empty name')
    return text


def _split(text: str, sep: str) -> list[str]:
    parts, start, at = [], 0, 0
    while at < len(text):
        if text[at] == '"':
            at = _quoted(text, at)[1]
        elif text[at] == sep:
            parts.append(text[start:at])
            at += 1
            start = at
        else:
            at += 1
    parts.append(text[start:])
    return parts


def _read_list(text: str, at: int) -> tuple[list, int]:
    items: list = []
    at += 1
    if text[at:at + 1] == ']':
        return items, at + 1
    while True:
        char = text[at:at + 1]
        if char == '[':
            item, at = _read_list(text, at)
        elif char == '"':
            item, at = _quoted(text, at)
        else:
            end = at
            while end < len(text) and text[end] not in ',]':
                end += 1
            item, at = _bare(text[at:end]), end
        items.append(item)
        char = text[at:at + 1]
        if char == ']':
            return items, at + 1
        if char != ',':
            raise ValueError(f'bad list {text!r}')
        at += 1


def _read_pairs(text: str) -> list[tuple[str, object]]:
    pairs = []
    for part in _split(text, ' '):
        if part.startswith('"'):
            key, end = _quoted(part)
        else:
            end = part.find('=')
            key = part[:end]
            if end < 1:
                raise ValueError(f'bad pair {part!r}')
        if part[end:end + 1] != '=':
            raise ValueError(f'bad pair {part!r}')
        pairs.append((key, _read_inline(part[end + 1:], False)))
    return pairs


def _read_inline(text: str, dicts: bool = True):
    if text.startswith('['):
        value, end = _read_list(text, 0)
        if end != len(text):
            raise ValueError(f'bad list {text!r}')
        return value
    if text.startswith('{') and dicts:
        if not text.endswith('}'):
            raise ValueError(f'bad dict {text!r}')
        return _unique(_read_pairs(text[1:-1])) if len(text) > 2 else {}
    if text.startswith('"'):
        return _token(text)
    return _bare(text)


def _unique(pairs) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('duplicate key')
    return result


def _copy(value):
    return [_copy(v) for v in value] if isinstance(value, list) else value


class _Reader:
    def __init__(self, lines: list[tuple[int, str]], reordered: list):
        self.lines = lines
        self.at = 0
        self.reordered = reordered

    def fail(self, message: str):
        raise ValueError(f'line {self.at + 1}: {message}')

    def line(self, indent: int) -> str | None:
        """Content of the current line when it sits exactly at `indent`."""
        if self.at < len(self.lines) and self.lines[self.at][0] == indent:
            return self.lines[self.at][1]
        return None

    def block(self, indent: int, item: bool = False):
        text = self.line(indent)
        if not text:
            self.fail('expected a value')
        try:
            if text.startswith(('- ', HOIST)):
                return self.items(indent)
            if text[0] == '[' and text.endswith(':'):
                return self.table(indent, text)
            if text[0] == '"':
                if _quoted(text)[1] != len(text):
                    return self.entries(indent)
            elif text[0] not in '[{' and _ENTRY_LINE.match(text):
                return self.entries(indent)
            self.at += 1
            value = _read_inline(text)
            if item and text[0] == '{' and value and self.line(indent) is not None:
                rest = self.entries(indent)
                if value.keys() & rest.keys():
                    self.fail('duplicate key')
                value.update(rest)
            return value
        except ValueError as exc:
            if str(exc).startswith('line '):
                raise
            self.fail(str(exc))

    def entries(self, indent: int) -> dict:
        result: dict = {}
        while (text := self.line(indent)) is not None:
            if text.startswith('"'):
                key, end = _quoted(text)
            else:
                colon, bracket = text.find(': '), text.find('[')
                if bracket != -1 and (colon == -1 or bracket < colon):
                    end = bracket
                elif colon != -1:
                    end = colon
                elif text.endswith(':'):
                    end = len(text) - 1
                else:
                    self.fail('expected key: value')
                key = text[:end]
                if _KEY.search(key):
                    self.fail(f'bad bare key {key!r}')
            rest = text[end:]
            if key in result:
                self.fail(f'duplicate key {key!r}')
            if rest.startswith(': '):
                self.at += 1
                result[key] = _read_inline(rest[2:])
            elif rest == ':':
                self.at += 1
                result[key] = self.block(indent + 2)
            elif rest.startswith('['):
                result[key] = self.table(indent, rest)
            else:
                self.fail('expected key: value')
        return result

    def hoisted(self, indent: int):
        text = self.line(indent)
        if text is None or not text.startswith(HOIST):
            return None
        match = _HOISTED.fullmatch(text)
        if not match:
            self.fail(f'bad {HOIST} line')
        self.at += 1
        return _read_pairs(match.group(1))

    def items(self, indent: int) -> list:
        hoisted = self.hoisted(indent)
        result = []
        while (text := self.line(indent)) is not None:
            if not text.startswith('- '):
                self.fail('expected a "- " item')
            self.lines[self.at] = (indent + 2, text[2:])
            result.append(self.block(indent + 2, True))
        if hoisted:
            if not result or not all(isinstance(row, dict) for row in result):
                self.fail(f'{HOIST} needs dict items')
            result = [_unique(list(row.items()) + [(k, _copy(v)) for k, v in hoisted]) for row in result]
            self.reordered += result
        return result

    def table(self, indent: int, header: str):
        match = _HEADER.fullmatch(header)
        if not match:
            self.fail('bad table header')
        names = _split(match.group(2), '|')
        keyed = names[0] == '*'
        columns = [_token(name) for name in names[keyed:]]
        if not columns:
            self.fail('table without columns')
        self.at += 1
        hoisted = self.hoisted(indent + 2)
        constants: dict = {}
        if hoisted:
            columns += [name for name, _ in hoisted]
            constants = _unique(hoisted)
        if len(set(columns)) != len(columns):
            self.fail('duplicate column')
        nested = list(dict.fromkeys(column.partition('.')[0] for column in columns if '.' in column))
        rows, keys = [], []
        while (text := self.line(indent + 2)) is not None:
            cells = _split(text, '|')
            if len(cells) != len(names):
                self.fail('wrong number of cells')
            if keyed:
                keys.append(_token(cells[0]))
            values = iter([_read_inline(cell, False) for cell in cells[keyed:]])
            row: dict = {}
            for column in columns:
                value = _copy(constants[column]) if column in constants else next(values)
                outer, dot, inner = column.partition('.')
                if dot:
                    target = row.setdefault(outer, {})
                    if not isinstance(target, dict) or inner in target:
                        self.fail(f'duplicate column {column!r}')
                    target[inner] = value
                elif outer in row:
                    self.fail(f'duplicate column {column!r}')
                else:
                    row[outer] = value
            rows.append(row)
            if hoisted:
                self.reordered.append(row)
                self.reordered += [row[outer] for outer in nested]
            self.at += 1
        if len(rows) != int(match.group(1)):
            self.fail('row count does not match the header')
        return _unique(list(zip(keys, rows))) if keyed else rows
