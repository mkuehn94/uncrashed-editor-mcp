import json
import random
import re
import unittest
from pathlib import Path
from unittest import mock

import uncrashed_mcp.render as render_module
from uncrashed_mcp.render import HOIST, parse, render

REPLIES = sorted((Path(__file__).parent / 'data' / 'replies').glob('*.json'))

AWKWARD = [
    '', ' x', 'x ', 'a|b', 'a,b', '[x', 'x]', '{', '}', '{}', '[]', '- x', '-', '-x', '#x', 'x#y',
    'true', 'false', 'null', 'True', 'NaN', 'Infinity', '-Infinity', '12', '-12', '1e5', '-0.0', '007', '+5', '.5',
    '1.', '1_000', '0x1F', 'k: v', 'k:', 'k: ', ':', ': x', 'a:b', 'two words: then prose', 'word: then prose',
    'ends with colon:', 'http://host/path', 'x=y', '=', 'a b', 'a.b', '.', '"q"', 'say "hi" twice', 'back\\slash',
    "it's", '*', '(every row also has) x=1', '(every row also has)', '(every row also has', '(every row',
    '(every row also has)x=1', '(all rows) x=1', '(all rows, keys 1) x=1', '(unnamed)', 't[2]{a|b}:', 'x[1]',
    '[2]{a}:', 'NewRow_788', '1a2b3c40', '12345678', '76561190000000000', 'MAP D | Night Run',
    'gr\u00fc\u00dfe', '\u65e5\u672c\u8a9e', '\U0001f600', 'line\nbreak', 'tab\there', '\r', '\x00', '\x7f',
    chr(0x85), chr(0x2028), chr(0x2029), chr(0xd800), chr(0xa0) + 'nbsp', 'trailing' + chr(0x3000),
    'x' * 31, 'cell ' * 17, 'a long sentence, with commas: and a colon. ' * 12,
]
NUMBERS = [0, 1, -1, 12, 10 ** 20, -(2 ** 63), 0.0, -0.0, 1.0, 1.5, 1e-07, 1e+20, 1e16, -2.5e-10, 123456.789,
           5e-324, 1.7976931348623157e+308]
PLAIN = ['id', 'name', 'kind', 'x', 'y', 'z', 'size_m', 'count', 'row', 'at', 'min', 'max']


def exact(value) -> str:
    """Text that differs for 1 / 1.0 / True, -0.0 / 0.0 and for a changed key order."""
    return json.dumps(value)


def same(value, back, loose=frozenset()) -> bool:
    """Type-exact equality (1 / 1.0 / True, -0.0 / 0.0 differ); key order counts except in dicts whose id is in `loose`."""
    if type(value) is not type(back):
        return False
    if isinstance(value, dict):
        if id(back) in loose:
            if value.keys() != back.keys():
                return False
        elif list(value) != list(back):
            return False
        return all(same(item, back[key], loose) for key, item in value.items())
    if isinstance(value, list):
        return len(value) == len(back) and all(same(a, b, loose) for a, b in zip(value, back))
    if isinstance(value, float):
        return repr(value) == repr(back)
    return value == back


def decode(text: str):
    """The decoded value and the ids of its dicts read under a hoist line (kept alive by the value)."""
    reordered: list = []
    back = parse(text, reordered)
    return back, {id(row) for row in reordered}


def without_hoisting(value) -> str:
    with mock.patch.object(render_module, 'HOIST_ROWS', 10 ** 9):
        return render(value)


class Fuzz:
    def __init__(self, seed: int):
        self.r = random.Random(seed)

    def key(self):
        return self.r.choice(PLAIN if self.r.random() < 0.7 else AWKWARD)

    def keys(self, low: int, high: int) -> list[str]:
        return list(dict.fromkeys(self.key() for _ in range(self.r.randint(low, high))))

    def scalar(self):
        pick = self.r.random()
        if pick < 0.1:
            return self.r.choice([None, True, False])
        if pick < 0.4:
            return self.r.choice(NUMBERS)
        if pick < 0.6:
            return self.r.choice(['mesh', 'House 02', 'NewRow_1', 'Big structure', 'ok'])
        return self.r.choice(AWKWARD)

    def cell(self):
        pick = self.r.random()
        if pick < 0.75:
            return self.scalar()
        if pick < 0.95:
            return [self.scalar() for _ in range(self.r.randint(0, 4))]
        return [[self.scalar() for _ in range(self.r.randint(0, 3))] for _ in range(self.r.randint(0, 3))]

    def rows(self, depth: int) -> list[dict]:
        columns = []
        for key in self.keys(1, 14):
            kind = self.r.random()
            if kind < 0.25:
                columns.append((key, 'constant', self.cell()))
            elif kind < 0.4:
                columns.append((key, 'flat', [(sub, self.r.random() < 0.3 and self.cell())
                                              for sub in self.keys(1, 3)]))
            else:
                columns.append((key, 'cell', None))
        rows = []
        for _ in range(self.r.randint(1, 8)):
            row = {}
            for key, kind, extra in columns:
                if kind == 'constant':
                    row[key] = extra
                elif kind == 'flat':
                    row[key] = {sub: fixed if fixed is not False else self.cell() for sub, fixed in extra}
                else:
                    row[key] = self.cell()
            rows.append(row)
        spoil = self.r.random()
        victim = self.r.choice(rows)
        if spoil < 0.08:
            victim.pop(next(iter(victim)))
        elif spoil < 0.16:
            victim[self.key()] = self.value(depth - 1)
        elif spoil < 0.24 and len(victim) > 1:
            first = next(iter(victim))
            victim[first] = victim.pop(first)
        return rows

    def value(self, depth: int):
        pick = self.r.random()
        if depth <= 0 or pick < 0.3:
            return self.cell()
        if pick < 0.5:
            return {key: self.value(depth - 1) for key in self.keys(0, 15)}
        if pick < 0.75:
            return self.rows(depth)
        if pick < 0.85:
            rows = self.rows(depth)
            return dict(zip(self.keys(len(rows), len(rows) + 2), rows))
        return [self.value(depth - 1) for _ in range(self.r.randint(0, 5))]


class RoundTripTests(unittest.TestCase):
    def check(self, value):
        text = render(value)
        try:
            back, loose = decode(text)
        except ValueError as exc:
            self.fail(f'{exc}\nvalue: {value!r}\ntext:\n{text}')
        if not same(value, back, loose) or not loose and exact(back) != exact(value):
            self.fail(f'round trip changed the value\nvalue: {value!r}\nback:  {back!r}\ntext:\n{text}')
        if HOIST not in text:
            self.assertFalse(loose)
        self.assertNotIn('\n\n', text)
        self.assertFalse(text.startswith('\n') or text.endswith('\n'))
        return text

    def test_replies_round_trip_and_never_exceed_compact_json(self):
        self.assertGreaterEqual(len(REPLIES), 44)
        json_total = text_total = 0
        for path in REPLIES:
            with self.subTest(reply=path.stem):
                value = json.loads(path.read_text(encoding='utf-8'))
                text = self.check(value)
                compact = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
                self.assertLessEqual(len(text), len(compact))
                self.assertLessEqual(len(text), len(without_hoisting(value)))
                json_total += len(compact)
                text_total += len(text)
        self.assertLess(text_total, 0.8 * json_total)

    def test_replies_carry_no_personal_data(self):
        for path in REPLIES:
            text = path.read_text(encoding='utf-8')
            with self.subTest(reply=path.stem):
                self.assertNotIn('_synthetic', text)
                self.assertNotRegex(text, r'/home/(?!user/)')
                self.assertNotRegex(text, r'(?<![0-9])(?!76561190000000000)[0-9]{17}(?![0-9])')

    def test_fuzzed_values(self):
        fuzz = Fuzz(20261120)
        for _ in range(4000):
            self.check(fuzz.value(3))

    def test_fuzzed_rows_and_keyed_rows(self):
        fuzz = Fuzz(7)
        for _ in range(1500):
            rows = fuzz.rows(2)
            self.check(rows)
            self.check({'rows': rows, 'by_key': dict(zip(fuzz.keys(len(rows), len(rows)), rows))})
            self.check([rows, {'nested': [rows]}])

    def test_hoisting_never_lengthens_a_table_or_an_item_list(self):
        fuzz, tables, lists, needed = Fuzz(11), 0, 0, 0
        for _ in range(3000):
            rows = fuzz.rows(1)
            for value in ({'t': rows}, {'t': dict(zip(fuzz.keys(len(rows), len(rows)), rows))}):
                text, plain = render(value), without_hoisting(value)
                if text == plain:
                    continue
                self.assertTrue(text.split('\n')[1].startswith('  ' + HOIST + ' '), text)
                if text.startswith('t:'):
                    lists += 1
                elif plain.startswith('t:'):  # a table only thanks to hoisting: kept whatever its length
                    needed += 1
                    continue
                else:
                    tables += 1
                self.assertLess(len(text), len(plain), f'\n{text}\n{plain}')
        self.assertGreater(min(tables, lists, needed), 20)

    def test_every_awkward_string_in_every_position(self):
        for text in AWKWARD:
            with self.subTest(text=text):
                self.check(text)
                self.check({text: text})
                self.check({'k': {text: text, 'n': 1}})
                self.check([text])
                self.check([text, 'x' * 40])
                self.check([[text, 1], [text]])
                self.check([{text: text, 'n': 1}, {text: text, 'n': 2}])
                self.check([{text: text, 'n': 1}, {text: text, 'n': 2}, {text: text, 'n': 3}])
                for count in (3, 4, 9):
                    self.check([{text: text, 'n': i} for i in range(count)])
                    self.check([{'a': 1, text: [text], 'n': i, 'deep': {'x': [1]}} for i in range(count)])
                    self.check([{'id': i, 'sub': {text: text, 'n': i}} for i in range(count)])
                    self.check([{text: text, 'id': i, 'wide': 'w' * 81 + str(i), text + 'x': [text]} for i in range(count)])
                    self.check({'d': {f'k{i}': {'v': i, 'c': text} for i in range(count)}})
                    self.check({'d': {text + str(i): {text: [text, text], 'v': i} for i in range(count)}})
                self.check({'d': {text: {'v': text, 'n': 1}, 'other': {'v': 'x', 'n': 2}}})
                self.check([{'a': 1, 'b': 2, text: {'x': {}}}, [text], text])

    def test_scalars_keep_their_type(self):
        for value in [None, True, False, 0, 1, -1, 1.0, -0.0, 0.0, 1e-07, 1e+20, 10 ** 30, '1', '1.0', 'true', 'null', '']:
            with self.subTest(value=value):
                self.check(value)
                self.check([value])
                self.check({'v': value})
                self.check([{'v': value, 'n': 1}, {'v': value, 'n': 2}, {'v': value, 'n': 3}])
                self.assertIn(HOIST, self.check([{'v': [value, value, value], 'n': i} for i in range(9)]))
        self.assertEqual(render({'a': 1, 'b': 1.0, 'c': -0.0, 'd': True, 'e': None, 'f': 1e-07, 'g': 1e+20}),
                         'a: 1\nb: 1.0\nc: -0.0\nd: true\ne: null\nf: 1e-07\ng: 1e+20')
        self.assertEqual(repr(parse('[-0.0,0.0,0,1e+20]')), '[-0.0, 0.0, 0, 1e+20]')
        self.assertEqual(repr(parse(render([float('nan'), float('inf'), float('-inf'), 'NaN']))),
                         "[nan, inf, -inf, 'NaN']")

    def test_empty_containers(self):
        self.assertEqual(render({}), '{}')
        self.assertEqual(render([]), '[]')
        self.assertEqual(render(''), '""')
        self.assertEqual(render({'issues': {}, 'top': [], 'note': '', 'rows': [{}, {}], 'cell': [[], {}]}),
                         'issues: {}\ntop: []\nnote: ""\nrows:\n  - {}\n  - {}\ncell:\n  - []\n  - {}')
        for value in [{}, [], [[]], [{}], {'a': {}}, {'a': []}, [[], []], [{}, []], {'a': {'b': {}}}]:
            self.check(value)

    def test_hoisted_keys_come_back_last_and_no_other_key_moves(self):
        rows = [{'z': 1, 'a': 'same', 'm': {'q': i, 'p': 0}, 'b': i, 'c': None} for i in range(4)]
        value = {'rows': rows, 'after': {'y': 1, 'x': [2]}}
        text = self.check(value)
        self.assertEqual(text, 'rows[4]{m.q|b}:\n'
                               '  (every row also has) z=1 a=same m.p=0 c=null\n'
                               '  0|0\n  1|1\n  2|2\n  3|3\n'
                               'after: {y=1 x=[2]}')
        reordered: list = []
        back = parse(text, reordered)
        self.assertEqual(list(back), ['rows', 'after'])
        self.assertEqual(list(back['after']), ['y', 'x'])
        self.assertEqual([list(row) for row in back['rows']], [['m', 'b', 'z', 'a', 'c']] * 4)
        self.assertEqual([list(row['m']) for row in back['rows']], [['q', 'p']] * 4)
        self.assertEqual([id(d) for d in reordered], [id(d) for row in back['rows'] for d in (row, row['m'])])
        items = [{'kind': 'mesh', 'id': i, 'sub': {'b': 1, 'a': 'x' * 81 + str(i)}} for i in range(4)]
        reordered = []
        back = parse(self.check(items), reordered)
        self.assertEqual([list(row) for row in back], [['id', 'sub', 'kind']] * 4)
        self.assertEqual([list(row['sub']) for row in back], [['b', 'a']] * 4)
        self.assertEqual([id(d) for d in reordered], [id(row) for row in back])
        shuffled = [{'a': 1, 'b': 2}, {'b': 3, 'a': 4}]
        self.assertEqual(self.check(shuffled), '- {a=1 b=2}\n- {b=3 a=4}')

    def test_round_trip_comparison_is_strict_outside_hoisted_rows(self):
        self.assertTrue(same({'a': [1, {'b': -0.0}]}, {'a': [1, {'b': -0.0}]}))
        for value, back in [(1, 1.0), (1, True), (0, False), (0.0, -0.0), ('1', 1), (None, False), ([1], [1, 1]),
                            ({'a': 1, 'b': 2}, {'b': 2, 'a': 1}), ({'a': 1}, {'a': 1, 'b': 2}), ({'a': {}}, {'a': []})]:
            with self.subTest(value=value, back=back):
                self.assertFalse(same(value, back))
                self.assertFalse(same([value], [back]))
        back, loose = decode('t[4]{a|n.x}:\n  (every row also has) b=1 n.y=2\n  1|1\n  2|2\n  3|3\n  4|4\nd: {p=1 q=[]}')
        self.assertEqual(len(loose), 8)
        good = {'t': [{'b': 1, 'n': {'y': 2, 'x': i}, 'a': i} for i in range(1, 5)], 'd': {'p': 1, 'q': []}}
        self.assertTrue(same(good, back, loose))
        self.assertFalse(same(good, back))
        self.assertFalse(same({'d': good['d'], 't': good['t']}, back, loose))
        self.assertFalse(same({**good, 'd': {'q': [], 'p': 1}}, back, loose))
        self.assertFalse(same({**good, 't': [{**row, 'b': 1.0} for row in good['t']]}, back, loose))
        self.assertFalse(same({**good, 't': [{**row, 'n': {'x': row['a']}} for row in good['t']]}, back, loose))

    def test_rejects_values_json_cannot_hold(self):
        for value in [{1: 'a'}, {'a': {2: 'b'}}, [{None: 1}, {None: 2}], (1, 2), {'a': {1, 2}}, b'x', {'a': object()}]:
            with self.subTest(value=value), self.assertRaises(TypeError):
                render(value)

    def test_decoder_rejects_malformed_text(self):
        for text in ['', 'a: 1\n\nb: 2', 'a: 1\n  b: 2', 'a: 1\na: 2', 'a:\n', 'a: [1,2', 'a: [1,,2]', 'a: {b}',
                     'a: {b=1 b=2}', 'a: "open', 't[3]{a|b}:\n  1|2', 't[1]{a|b}:\n  1|2|3', 't[1]{a|b}:\n  1',
                     't[2]{a}:\n  (every row also has, keys 2) b=1\n  1\n  2', 't[2]{a}:\n  (every row also has)\n  1\n  2',
                     't[2]{a}:\n  (every row also has) a=1\n  1\n  2', 't[2]{a}:\n  (every row also has) b=1 b=2\n  1\n  2',
                     't[2]{a}:\n  (every row also has)b=1\n  1\n  2', 't[2]{a}:\n  1\n  (every row also has) b=1\n  2',
                     '(every row also has) a=1\n- 1\n- 2', '(every row also has) a=1\n- {a=2}', '(every row also has) a=1',
                     '- a\nb: 1', 'a b: 1\nc: 2', ' a: 1', 't[1]{a.b|a.b}:\n  1|2', 'a: 1\n- x', 'a: {b=1}\n  c: 2',
                     '- [1]\n  c: 2', 'a:  1', 'a: #1', '[1, 2]']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse(text)

    def test_decoder_reads_the_documented_forms(self):
        self.assertEqual(parse('a b: 1'), 'a b: 1')
        self.assertEqual(parse('a: 1\n'), {'a': 1})
        self.assertEqual(parse('- {a=1 b=2}\n  c:\n    - x\n  d: e: f\n- - 1\n  - [2]\n'),
                         [{'a': 1, 'b': 2, 'c': ['x'], 'd': 'e: f'}, [1, [2]]])
        self.assertEqual(exact(parse('t[2]{*|n|p.x}:\n  (every row also has) p.y=[0,"a|b"] q=1.0\n  k|1|2\n  "1"|3|4')),
                         exact({'t': {'k': {'n': 1, 'p': {'x': 2, 'y': [0, 'a|b']}, 'q': 1.0},
                                      '1': {'n': 3, 'p': {'x': 4, 'y': [0, 'a|b']}, 'q': 1.0}}}))
        self.assertEqual(exact(parse('(every row also has) kind=mesh "a b"=[1]\n- {id=1}\n- {id=2}\n  c:\n    - x')),
                         exact([{'id': 1, 'kind': 'mesh', 'a b': [1]}, {'id': 2, 'c': ['x'], 'kind': 'mesh', 'a b': [1]}]))
        rows = parse('[3]{a}:\n  (every row also has) tags=[x,[y]]\n  1\n  2\n  3')
        rows[0]['tags'][1].append('z')
        self.assertEqual(rows[1], {'a': 2, 'tags': ['x', ['y']]})

    def test_large_table_round_trips(self):
        rows = [{'id': f'NewRow_{i}', 'kind': 'mesh', 'at': [i * 0.5, -i, 0.0], 'aabb': {'min': [i], 'max': [i + 1]},
                 'hidden': i % 7 == 0} for i in range(3000)]
        text = self.check({'assets': rows})
        self.assertEqual(len(text.split('\n')), 3002)
        self.assertLess(len(text), 0.5 * len(json.dumps(rows, separators=(',', ':'))))


class GoldenTests(unittest.TestCase):
    def golden(self, value, *lines: str):
        text = render(value)
        self.assertEqual(text, '\n'.join(lines))
        back, loose = decode(text)
        self.assertTrue(same(value, back, loose))
        self.assertEqual(bool(loose), any(re.match(r' *(?:- )*' + re.escape(HOIST) + ' ', line) for line in lines))

    def test_table_hoists_constant_columns(self):
        common = {'kind': 'mesh', 'triplanar': False, 'material': None}
        rows = [('NewRow_788', 'House 02', 'Big structure', [9.08, 14.5, 8.28]),
                ('NewRow_790', 'House 04', 'Big structure', [11.87, 11.08, 6.8]),
                ('NewRow_337', 'Logcabin Roof', 'Wall', [4.0, 2.08, 0.07]),
                ('NewRow_4', 'Neons Pink', 'Interractive', [6.32, 1.79, 0.1])]
        assets = [{'id': i, 'name': n, 'category': c, 'kind': 'mesh', 'size_m': s, 'triplanar': False, 'material': None}
                  for i, n, c, s in rows]
        self.golden({'count': 4, 'assets': assets},
                    'count: 4',
                    'assets[4]{id|name|category|size_m}:',
                    '  (every row also has) kind=mesh triplanar=false material=null',
                    '  NewRow_788|House 02|Big structure|[9.08,14.5,8.28]',
                    '  NewRow_790|House 04|Big structure|[11.87,11.08,6.8]',
                    '  NewRow_337|Logcabin Roof|Wall|[4.0,2.08,0.07]',
                    '  NewRow_4|Neons Pink|Interractive|[6.32,1.79,0.1]')
        self.golden({'t': [{'a': i, **common} for i in range(4)]},
                    't[4]{a}:', '  (every row also has) kind=mesh triplanar=false material=null', '  0', '  1', '  2', '  3')

    def test_hoisting_needs_three_rows_and_a_remaining_column(self):
        suggestions = [{'id': f'NewRow_{i}', 'name': 'Neons02', 'kind': 'mesh', 'tags': ['sign', 'neon'], 'hidden': i > 37,
                        'score': 7.91} for i in (37, 372, 373)]
        self.golden({'suggestions': suggestions},
                    'suggestions[3]{id|hidden}:',
                    '  (every row also has) name=Neons02 kind=mesh tags=[sign,neon] score=7.91',
                    '  NewRow_37|false',
                    '  NewRow_372|true',
                    '  NewRow_373|true')
        self.golden({'suggestions': suggestions[:2]},
                    'suggestions[2]{id|name|kind|tags|hidden|score}:',
                    '  NewRow_37|Neons02|mesh|[sign,neon]|false|7.91',
                    '  NewRow_372|Neons02|mesh|[sign,neon]|true|7.91')
        self.golden({'t': [{'a': i, 'b': 'x'} for i in range(2)]}, 't[2]{a|b}:', '  0|x', '  1|x')
        self.golden({'t': [{'a': 1, 'b': 'x'}] * 4}, 't[4]{a|b}:', '  1|x', '  1|x', '  1|x', '  1|x')
        self.golden({'t': [{'a': 1, 'b': 'x'}]}, 't:', '  - {a=1 b=x}')

    def test_hoisting_only_when_it_shortens_the_table(self):
        def examples(count, gap=None):
            return {'examples': [{'x': i * 1.5, 'gap_m': i if gap is None else gap, 'bottom': -0.5} for i in range(count)]}
        self.golden(examples(5), 'examples[5]{x|gap_m|bottom}:',
                    '  0.0|0|-0.5', '  1.5|1|-0.5', '  3.0|2|-0.5', '  4.5|3|-0.5', '  6.0|4|-0.5')
        self.golden(examples(6), 'examples[6]{x|gap_m}:', '  (every row also has) bottom=-0.5',
                    '  0.0|0', '  1.5|1', '  3.0|2', '  4.5|3', '  6.0|4', '  7.5|5')
        self.golden(examples(4, 1.25), 'examples[4]{x}:',
                    '  (every row also has) gap_m=1.25 bottom=-0.5', '  0.0', '  1.5', '  3.0', '  4.5')
        def nested(count):  # at 6 rows both forms have the same length: not hoisted
            return {'a': {'t': [{'id': i, 'kind': 'mesh'} for i in range(count)]}}
        self.golden(nested(6), 'a:', '  t[6]{id|kind}:', *(f'    {i}|mesh' for i in range(6)))
        self.golden(nested(7), 'a:', '  t[7]{id}:', '    (every row also has) kind=mesh', *(f'    {i}' for i in range(7)))
        self.assertEqual(len(render(nested(6))), len(render(nested(6)).replace('|kind', '').replace('|mesh', ''))
                         + len('\n    (every row also has) kind=mesh'))
        for count in range(2, 12):
            for gap in (None, 1, 1.25):
                with self.subTest(count=count, gap=gap):
                    value = examples(count, gap)
                    self.assertLessEqual(len(render(value)), len(without_hoisting(value)))
                    self.assertLessEqual(len(render({'in': value})), len(without_hoisting({'in': value})))

    def test_long_cell_falls_back_to_one_line_rows(self):
        long = 'anchor bottom_center puts the bounding-box centre on x,y and the bottom on z; the pivot is a corner'
        self.assertGreater(len(long), 80)
        assets = [{'id': 'NewRow_1', 'kind': 'mesh', 'placement': long, 'size_m': [1.0, 2.0, 3.0]},
                  {'id': 'NewRow_2', 'kind': 'mesh', 'placement': 'pivot = base', 'size_m': [4.0, 5.0, 6.0]},
                  {'id': 'NewRow_3', 'kind': 'mesh', 'placement': long, 'size_m': [7.0, 8.0, 9.0]},
                  {'id': 'NewRow_4', 'kind': 'mesh', 'placement': 'pivot = base', 'size_m': [1.0, 1.0, 1.0]}]
        self.golden({'assets': assets},
                    'assets:',
                    '  (every row also has) kind=mesh',
                    f'  - {{id=NewRow_1 placement="{long}" size_m=[1.0,2.0,3.0]}}',
                    '  - {id=NewRow_2 placement="pivot = base" size_m=[4.0,5.0,6.0]}',
                    f'  - {{id=NewRow_3 placement="{long}" size_m=[7.0,8.0,9.0]}}',
                    '  - {id=NewRow_4 placement="pivot = base" size_m=[1.0,1.0,1.0]}')
        self.golden({'assets': assets[:3]},
                    'assets:',
                    f'  - {{id=NewRow_1 kind=mesh placement="{long}" size_m=[1.0,2.0,3.0]}}',
                    '  - {id=NewRow_2 kind=mesh placement="pivot = base" size_m=[4.0,5.0,6.0]}',
                    f'  - {{id=NewRow_3 kind=mesh placement="{long}" size_m=[7.0,8.0,9.0]}}')
        self.golden([{'id': i, 'kind': 'x', 'text': 'x' * 81 + str(i)} for i in range(4)],  # same length both ways
                    *(f'- {{id={i} kind=x text={"x" * 81}{i}}}' for i in range(4)))
        self.golden([{'id': i, 'kind': 'xy', 'text': 'x' * 81 + str(i)} for i in range(4)],
                    '(every row also has) kind=xy', *(f'- {{id={i} text={"x" * 81}{i}}}' for i in range(4)))
        self.golden([{'id': i, 'k': 1, 'sub': {'a': 'x' * 81 + str(i)}} for i in range(5)],  # same length both ways
                    *(line for i in range(5) for line in (f'- {{id={i} k=1}}', f'  sub: {{a={"x" * 81}{i}}}')))
        self.golden([{'id': i, 'kind': 'x', 'sub': {'a': 'x' * 81 + str(i)}} for i in range(4)],
                    '(every row also has) kind=x',
                    *(line for i in range(4) for line in (f'- id: {i}', f'  sub: {{a={"x" * 81}{i}}}')))
        rows = value = [{'id': i, 'text': 'x' * 81 + str(i), 'tags': ['y' * 31, 'z']} for i in range(4)]
        self.golden(rows, f'(every row also has) tags=[{"y" * 31},z]', *(f'- {{id={i} text={"x" * 81}{i}}}' for i in range(4)))
        for _ in range(60):  # the three `tags` lines each item loses outweigh the long indent of the hoist line
            value = [value]
        text = render(value)
        self.assertEqual(text.split('\n')[0], '- ' * 60 + f'(every row also has) tags=[{"y" * 31},z]')
        self.assertTrue(same(value, *decode(text)))
        self.assertLess(len(text), len(without_hoisting(value)))
        self.golden({'t': [{'id': 1, 'text': 'x' * 80}, {'id': 2, 'text': 'y'}]},
                    't[2]{id|text}:', '  1|' + 'x' * 80, '  2|y')
        self.golden({'t': [{'id': 1, 'text': 'x' * 81}, {'id': 2, 'text': 'y'}]},
                    't:', '  - {id=1 text=' + 'x' * 81 + '}', '  - {id=2 text=y}')

    def test_wide_table_falls_back_to_one_line_rows(self):
        self.golden({'wide': [{f'c{j}': i * j for j in range(11)} for i in range(2)]},
                    'wide:',
                    '  - {c0=0 c1=0 c2=0 c3=0 c4=0 c5=0 c6=0 c7=0 c8=0 c9=0 c10=0}',
                    '  - {c0=0 c1=1 c2=2 c3=3 c4=4 c5=5 c6=6 c7=7 c8=8 c9=9 c10=10}')
        self.golden({'wide': [{**{f'c{j}': (i + 1) * j for j in range(10)}, 'unit': 'cm'} for i in range(2)]},
                    'wide:',  # 11 columns and too few rows to hoist any
                    '  - {c0=0 c1=1 c2=2 c3=3 c4=4 c5=5 c6=6 c7=7 c8=8 c9=9 unit=cm}',
                    '  - {c0=0 c1=2 c2=4 c3=6 c4=8 c5=10 c6=12 c7=14 c8=16 c9=18 unit=cm}')

    def test_hoisting_stays_when_only_it_makes_the_table_fit(self):
        self.golden({'wide': [{**{f'c{j}': i * j for j in range(10)}, 'u': 1} for i in range(4)]},
                    'wide[4]{c1|c2|c3|c4|c5|c6|c7|c8|c9}:',
                    '  (every row also has) c0=0 u=1',
                    '  0|0|0|0|0|0|0|0|0',
                    '  1|2|3|4|5|6|7|8|9',
                    '  2|4|6|8|10|12|14|16|18',
                    '  3|6|9|12|15|18|21|24|27')
        self.golden({'t': [{'id': i, 'note': 'n' * 81} for i in range(4)]},
                    't[4]{id}:', '  (every row also has) note=' + 'n' * 81, '  0', '  1', '  2', '  3')

    def test_flat_dicts_become_dotted_columns(self):
        groups = [{'name': 'house.walls', 'objects': 2, 'pivot_bounds_m': {'min': [123.5, -44.1, 0.1], 'max': [129.5, -44.1, 0.1]}},
                  {'name': '-', 'objects': 1, 'pivot_bounds_m': {'min': [0.0, 0.0, 0.0], 'max': [0.0, 0.0, 0.0]}}]
        self.golden({'groups': groups},
                    'groups[2]{name|objects|pivot_bounds_m.min|pivot_bounds_m.max}:',
                    '  house.walls|2|[123.5,-44.1,0.1]|[129.5,-44.1,0.1]',
                    '  -|1|[0.0,0.0,0.0]|[0.0,0.0,0.0]')
        self.golden({'top': [{'id': i, 'view': {'x': i * 1.5, 'yaw': 45.0, 'pitch': -24.2}} for i in range(4)]},
                    'top[4]{id|view.x}:',
                    '  (every row also has) view.yaw=45.0 view.pitch=-24.2',
                    '  0|0.0', '  1|1.5', '  2|3.0', '  3|4.5')
        self.golden({'top': [{'id': i, 'view': {'x': i * 1.5, 'yaw': 45.0, 'pitch': -24.2}} for i in range(3)]},
                    'top[3]{id|view.x|view.yaw|view.pitch}:',
                    '  0|0.0|45.0|-24.2', '  1|1.5|45.0|-24.2', '  2|3.0|45.0|-24.2')

    def test_dotted_keys_and_ragged_rows_stay_out_of_tables(self):
        self.golden({'rows': [{'r.VSync': 0.0, 'n': 1}, {'r.VSync': 1.0, 'n': 2}]},
                    'rows:', '  - {r.VSync=0.0 n=1}', '  - {r.VSync=1.0 n=2}')
        self.golden({'rows': [{'a': 1, 'b': 2}, {'a': 3}, {'a': 4, 'b': None, 'skipped': 'gate'}]},
                    'rows:', '  - {a=1 b=2}', '  - {a=3}', '  - {a=4 b=null skipped=gate}')
        self.golden({'rows': [{'a': 1, 'view': {'x': 1}}, {'a': 2, 'view': None}]},
                    'rows:', '  - a: 1', '    view: {x=1}', '  - {a=2 view=null}')

    def test_keyed_table(self):
        self.golden({'budgets': {'placed_objects': {'warn': 4000, 'high': 10000},
                                 'triangles': {'warn': 8000000.0, 'high': 25000000.0},
                                 'frame_ms_avg': {'warn': 16.7, 'high': 33.3}}},
                    'budgets[3]{*|warn|high}:',
                    '  placed_objects|4000|10000',
                    '  triangles|8000000.0|25000000.0',
                    '  frame_ms_avg|16.7|33.3')
        self.golden({'zones': {'court': {'yaw': 45.0, 'size': [4, 6], 'partial': False},
                               'a|b': {'yaw': -0.0, 'size': [2, 2], 'partial': False},
                               '12': {'yaw': 0.0, 'size': [1, 1], 'partial': False}}},
                    'zones[3]{*|yaw|size|partial}:',
                    '  court|45.0|[4,6]|false',
                    '  "a|b"|-0.0|[2,2]|false',
                    '  "12"|0.0|[1,1]|false')
        self.golden({'zones': {name: {'yaw': yaw, 'size': [4, 6], 'partial': False, 'source': 'base_instance'}
                               for name, yaw in [('court', 45.0), ('a|b', -0.0), ('12', 0.0), ('*', 90.0)]}},
                    'zones[4]{*|yaw}:',
                    '  (every row also has) size=[4,6] partial=false source=base_instance',
                    '  court|45.0',
                    '  "a|b"|-0.0',
                    '  "12"|0.0',
                    '  *|90.0')
        self.golden({'list': [{'*': 1, 'n': 2}, {'*': 3, 'n': 4}]}, 'list[2]{"*"|n}:', '  1|2', '  3|4')
        self.golden({'one': {'only': {'warn': 1, 'high': 2}}}, 'one:', '  only: {warn=1 high=2}')

    def test_inline_dict(self):
        self.golden({'camera': {'x': -63.9, 'y': -13.9, 'z': 33.0, 'yaw': 45.0, 'pitch': -31.0, 'held': False, 'backend': 'native'},
                     'per_category': {'Ground': 1600, 'Big structure': 17},
                     'probe': {'axes': 'left/right, up/down', 'expr': 'a=b', 'set': '{x}', 'none': None, 'id': '42',
                               'size': [1, 'two words', 'a,b']},
                     'nested': {'a': {'b': 1}}},
                    'camera: {x=-63.9 y=-13.9 z=33.0 yaw=45.0 pitch=-31.0 held=false backend=native}',
                    'per_category: {Ground=1600 "Big structure"=17}',
                    'probe: {axes="left/right, up/down" expr="a=b" set="{x}" none=null id="42" size=[1,"two words","a,b"]}',
                    'nested:',
                    '  a: {b=1}')
        self.golden({'d': {f'k{i}': i for i in range(12)}}, 'd: {' + ' '.join(f'k{i}={i}' for i in range(12)) + '}')
        self.golden({'d': {f'k{i}': i for i in range(13)}}, 'd:', *(f'  k{i}: {i}' for i in range(13)))
        self.golden({'camera': {'x': 1}}['camera'], 'x: 1')

    def test_long_string_list(self):
        self.golden({'tags': ['building', 'house'],
                     'limitations': ['Read-only heuristic diagnostics, not a flight test.',
                                     'Note: quoted because the first word ends with a colon',
                                     'Loaded geometry only: empty space counts.', 'short'],
                     'thirty': ['x' * 30, 'y'],
                     'zones': {'native': ['zones count=0 masked=0 restored=0 unavailable=0']}},
                    'tags: [building,house]',
                    'limitations:',
                    '  - Read-only heuristic diagnostics, not a flight test.',
                    '  - "Note: quoted because the first word ends with a colon"',
                    '  - Loaded geometry only: empty space counts.',
                    '  - short',
                    'thirty: [' + 'x' * 30 + ',y]',
                    'zones:',
                    '  native:',
                    '    - zones count=0 masked=0 restored=0 unavailable=0')

    def test_quoting_of_cells_and_numeric_looking_strings(self):
        maps = [{'folder': 'Map_D', 'name': 'MAP D | Night Run', 'steam_id': '76561190000000000', 'actor': '1a2b3c40',
                 'z': -0.0, 'laps': None},
                {'folder': 'Map_E', 'name': '2049', 'steam_id': '76561190000000001', 'actor': '12345678', 'z': 0.0, 'laps': 2},
                {'folder': 'Map F', 'name': ' padded', 'steam_id': '', 'actor': '17e63a10', 'z': 1e-07, 'laps': 3}]
        self.golden({'maps': maps, 'steam_id': '76561190000000000', 'actor': '12345678', 'hex': '1a2b3c40',
                     'flag': 'true', 'none': 'null', 'title': 'MAP D | Night Run',
                     'ids': ['12', 'x', 12, '1e5', 'a,b', '[x'], 'dash': '- x', 'hash': '#1', 'colon': 'k: v',
                     'text': 'say "hi"', 'lines': 'a\nb'},
                    'maps[3]{folder|name|steam_id|actor|z|laps}:',
                    '  Map_D|"MAP D | Night Run"|"76561190000000000"|1a2b3c40|-0.0|null',
                    '  Map_E|"2049"|"76561190000000001"|"12345678"|0.0|2',
                    '  Map F|" padded"|""|17e63a10|1e-07|3',
                    'steam_id: "76561190000000000"',
                    'actor: "12345678"',
                    'hex: 1a2b3c40',
                    'flag: "true"',
                    'none: "null"',
                    'title: MAP D | Night Run',
                    'ids: ["12",x,12,"1e5","a,b","[x"]',
                    'dash: "- x"',
                    'hash: "#1"',
                    'colon: k: v',
                    'text: say "hi"',
                    'lines: "a\\nb"')

    def test_unprintable_characters_are_escaped(self):
        odd = [chr(0x2028), chr(0xd800), chr(0x85), chr(0x7f), '\x00']
        text = render({'odd': odd, 'plain': 'gr' + chr(0xfc) + chr(0xdf) + 'e ' + chr(0x1f600)})
        self.assertEqual(text, 'odd: [' + ','.join(json.dumps(c) for c in odd) + ']\n'
                               'plain: gr' + chr(0xfc) + chr(0xdf) + 'e ' + chr(0x1f600))
        text.encode('utf-8')
        self.assertEqual(text.splitlines(), text.split('\n'))

    def test_key_quoting(self):
        self.golden({'Big structure': 1, 'a:': 2, 'x[1]': 3, '': 4, '12': 5, 'a:b': 6, 'r.VSync': 7, 'a|b': 8},
                    '"Big structure": 1', '"a:": 2', '"x[1]": 3', '"": 4', '"12": 5', 'a:b: 6', 'r.VSync: 7', 'a|b: 8')
        self.golden({'t': [{'a|b': i, 'c d': -i, 'e=f': 'same in all', 'g{': '(every row'} for i in range(4)]},
                    't[4]{"a|b"|c d}:', '  (every row also has) "e=f"="same in all" "g{"="(every row"',
                    '  0|0', '  1|-1', '  2|-2', '  3|-3')
        self.golden({'t': [{'a|b': i, 'e=f': 'x y', 'g{': '(every row also has) k=v'} for i in range(2)]},
                    't[2]{"a|b"|e=f|"g{"}:', *(f'  {i}|x y|"(every row also has) k=v"' for i in range(2)))
        self.golden({'note': '(every row also has) k=v', 'old': '(all rows) k=v', 'list': ['(every row also has)', '(every row']},
                    'note: "(every row also has) k=v"', 'old: (all rows) k=v', 'list: ["(every row also has)",(every row]')

    def test_list_items(self):
        issues = [{'id': 1, 'kind': 'collision', 'at_m': [9.8, 1.9, 1.5], 'obstacle': {'actor': 'w1', 'row': 'NewRow_7'},
                   'view': {'x': 6.65, 'yaw': -6.8}, 'gate': 1},
                  {'id': 2, 'kind': 'tight', 'at_m': [18.2, 6.1, 3.0]},
                  {'cells': [{'row': 'NewRow_181', 'count': 90}, {'row': 'NewRow_182', 'count': 46}], 'objects': 216},
                  {'only': {'deep': {'er': 1}}}, 'plain', 'key: value', ['a', 'b'], [{'a': 1}], None]
        self.golden({'issues': issues},
                    'issues:',
                    '  - {id=1 kind=collision at_m=[9.8,1.9,1.5]}',
                    '    obstacle: {actor=w1 row=NewRow_7}',
                    '    view: {x=6.65 yaw=-6.8}',
                    '    gate: 1',
                    '  - {id=2 kind=tight at_m=[18.2,6.1,3.0]}',
                    '  - cells[2]{row|count}:',
                    '      NewRow_181|90',
                    '      NewRow_182|46',
                    '    objects: 216',
                    '  - only:',
                    '      deep: {er=1}',
                    '  - plain',
                    '  - "key: value"',
                    '  - [a,b]',
                    '  - - {a=1}',
                    '  - null')

    def test_item_fields_sit_two_columns_right_of_their_dash(self):
        item = {'id': 1, 'kind': 'tight', 'obstacle': {'deep': {'er': 1}}, 'gate': 1}
        self.golden([item, [item], {'in': [item, 'x']}],
                    '- {id=1 kind=tight}',
                    '  obstacle:',
                    '    deep: {er=1}',
                    '  gate: 1',
                    '- - {id=1 kind=tight}',
                    '    obstacle:',
                    '      deep: {er=1}',
                    '    gate: 1',
                    '- in:',
                    '    - {id=1 kind=tight}',
                    '      obstacle:',
                    '        deep: {er=1}',
                    '      gate: 1',
                    '    - x')
        for path in REPLIES:
            lines = render(json.loads(path.read_text(encoding='utf-8'))).split('\n')
            for at, line in enumerate(lines[:-1]):
                dash, after = len(line) - len(line.lstrip(' ')), len(lines[at + 1]) - len(lines[at + 1].lstrip(' '))
                if line.lstrip(' ').startswith('- {') and after > dash:
                    self.assertEqual(after, dash + 2, f'{path.stem}: {lines[at + 1]}')

    def test_top_level_values(self):
        self.golden([{'x': 0, 'y': 0, 'z': 12.5}, {'x': 10, 'y': 0, 'z': None}], '[2]{x|y|z}:', '  0|0|12.5', '  10|0|null')
        self.golden(['a', 1, None], '[a,1,null]')
        self.golden('k: v', '"k: v"')
        self.golden('plain text, with: punctuation', 'plain text, with: punctuation')
        self.golden(-0.0, '-0.0')
        self.golden({'a': {'p': {'v': 1}, 'q': {'v': 2}}}['a'], 'p: {v=1}', 'q: {v=2}')


if __name__ == '__main__':
    unittest.main()
