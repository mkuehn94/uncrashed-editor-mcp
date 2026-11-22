# Command line

`./uc` drives the same 41 operations as the MCP server from a shell. Agents load nothing up front: `./uc help` lists the commands, `./uc help <command>` gives parameters and examples, `./uc help guide` carries the workflow and cautions. Replies are compact text instead of JSON, bulk input comes from files, images land on disk.

## Install

```bash
./uc help                      # from a checkout; uses .venv/bin/python if present, else python3
.venv/bin/pip install -e .     # also installs the `uncrashed` command, same as ./uc
```

`./uc` is a small sh launcher that puts `src/` on `PYTHONPATH`. Everything below says `./uc`; after `pip install -e .` the same line works as `uncrashed`. The CLI needs only Pillow. The `mcp` package is needed just for `./uc mcp` and `uncrashed-mcp` (`.venv/bin/pip install -e '.[mcp]'`); without it both print a one-line hint.

## Grammar

```
./uc [--json] [--pick PATHS] <command> [positionals] [--param value ...]
```

- Command names take `-` or `_` (`list-gates`), plus aliases: `exec`, `batch`, `log`, `unload`. A typo exits 2 with the nearest name.
- Positionals bind to the first parameters in order (`./uc move_camera 10 20 30 90`). Everything else is `--name value`; names are exact, no abbreviations. Surplus positionals and `name=value` tokens are usage errors.
- `-5`, `-.5` and `-` are values, not options. `--name=value` is the way to pass a value starting with `--`; `--` ends option parsing.
- Booleans: `--flag`, `--no-flag`, `--flag=false`. Edits with a `preview` parameter (marked `~` in `./uc help`) preview by default; `--apply` is `--no-preview`. `exec`, `batch`, `load_live`, `save_live` and `unload` (`!`) have no preview: they act at once. `--apply=0`, `--apply no` and the like are refused. After an `exec` command word every `--word` needs a leading `--` (`./uc exec -- console r.foo --bar`).
- Lists are one comma-separated token (`--delta 0,40,0`, `--actors 1A2B,3C4D`), a quoted JSON list, `@file` (whitespace, comma or newline separated) or `@-` for stdin. Repeating the option appends. `--rows=[]` is the explicit empty list.
- Points are `x,y[,z]` tokens: `./uc sample_terrain 10,20 12.5,20`; `--points @file` reads one per line. `inspect_site` takes two `x,y,z` tokens.
- Maps and JSON values: `--roles base=foundation,frame=structure`, or one JSON token in single quotes, or `@file`.
- `--region x0,y0,x1,y1` expands to the four region parameters on every command that has them.
- `exec` joins everything after the command word verbatim, so options go first: `./uc exec --max_lines 100 actors 0 0 0 500`. Quote anything the shell would eat (`* ! # ; | < > &`). One command per exec; more than one is `batch`.
- Units: commands take metres and degrees; `exec` / `batch` command text takes centimetres.

```bash
./uc search_assets house --limit 3
./uc capture_view 10 20 30 90 -10 --out shots/
./uc edit_group houses --action clone --target_group houses_b --delta 0,40,0 --apply
./uc check_collisions --region -100,-100,100,100
./uc exec mapinfo
```

## Output

Default output is a compact, lossless text form (`render.py` documents the grammar; `render.parse` reads it back). A real reply from `./uc search_assets house --limit 4`:

```
count: 4
assets[4]{id|name|size_m|long_axis}:
  (every row also has) category="Big structure" kind=mesh tags=[building,house] hidden=false pivot=offset triplanar=false material=null
  NewRow_788|House 02|[9.08,14.5,8.28]|y
  NewRow_790|House 04|[11.87,11.08,6.8]|x
  NewRow_789|House 08|[11.88,14.64,8.28]|y
  NewRow_791|House 10|[13.27,10.72,6.78]|x
```

`key: value` lines; `assets[4]{...}` is a table of 4 rows whose columns are listed in braces; values shared by every row (tables of 3 or more rows) are written once on the `(every row also has)` line, and that line puts those fields after the row's own columns, so key order inside such rows may differ from `--json`; a result that does not fit a table prints `- {a=1 b=2}` items, with the same line before them and `key: value` lines indented under an item as more fields of that item; `[a,b]` is a list; `{a=1 b=2}` an inline dict; strings are quoted only when they would be ambiguous. `./uc help output` has the full key. `exec` / `batch` / `log` print the agent's text; a failure shows as exit code 3 plus the stderr line, and an empty reply prints `(empty reply)` on stderr.

`--json` prints exactly one JSON document instead (`{"ok":..,"text":..}` for `exec` / `batch` / `log`, `{"error":..}` on failure; NaN / Infinity become `null` with a stderr warning). `UNCRASHED_FORMAT=json` makes that the default. `--pick a,b.c` keeps only those fields, mapped over lists:

```
$ ./uc search_assets house --limit 4 --pick assets.id,assets.name
assets[4]{id|name}:
  NewRow_788|House 02
  NewRow_790|House 04
  NewRow_789|House 08
  NewRow_791|House 10
```

A path that matches nothing prints a warning to stderr and the full result. Previews end with `preview only: nothing changed (re-run with --apply)` on stderr. `capture_view` writes a PNG and prints its path and metadata; `--out file.png` or `--out dir/` chooses where.

## Exit codes

| code | meaning |
| --- | --- |
| 0 | ok |
| 1 | the operation raised. `... nothing was sent` means nothing reached the game: safe to retry. For live changes after a send the scene may be partly changed |
| 2 | usage error, nothing ran |
| 3 | ran, but reported failures (some batch lines failed, `save_live` did not write, a `capture_sheet` pose failed, ...) |

Every non-zero exit also prints one `error: ...` line on stderr, even if stdout was printed, so a pipe cannot hide it. Use `set -o pipefail` in pipelines. An error that ends `may have run`, a timeout, or `batch state UNKNOWN` means a command was sent: never re-send it, look first (`./uc last`, `./uc exec mapinfo`). `nothing was sent` is the only safe-to-retry text.

## Batch workflow

Generate the command file with a script that lives in your project (not `/tmp`), then send it in one game-thread pass:

```bash
set -o pipefail
python3 plan.py > plan.txt
./uc batch plan.txt          # or: python3 plan.py | ./uc batch -
```

One command per line, centimetres; empty lines and `#` lines are skipped. Each call writes its own file `/tmp/uncrashed-agent-batch.<pid>.<n>.<token>.txt`, removes it once the job answered, leaves it when the command was sent without a reply, and sweeps ones older than a day. The first reply line is the agent summary plus `sent=N`. Failed lines (at most 20) are listed as `#<index> [L<file line>] <error> << <command>`, with a count of the rest; run `./uc last` before the next agent command to list them all. The exit code is 3; every other line **is applied**, so fix and resend only the failed ones. `--no-quiet` also returns every reply. `drop last` belongs on the line directly after its `spawn` in the same file.

## Long commands, timeouts, retries

`check_collisions` (about a minute per 27,000 objects), `load_live`, large batches, `consolidate_scenery --apply`, `check_route_clearance` and `check_opening` can outlast a shell tool's default timeout: raise it or run in the background. `--timeout` (seconds) sets how long the CLI waits for the agent's reply (`exec` default 30). `batch` waits max(60 s, 0.02 s per command) unless `--timeout` is given. A timeout means the command was sent and may still run. Never re-send it: look first. A batch that fails after sending ends `batch state UNKNOWN` (same rule). Follow-up live commands wait for the running job (note `agent is finishing an earlier job`) and give up with `agent busy: an earlier job is still running; nothing was sent` plus a hint to read `./uc log`.

```bash
./uc last              # the last finished agent reply; while a job runs it is the one before it (stderr note)
./uc exec mapinfo
./uc edit_history
./uc log 50
```

## Concurrency

One process talks to the agent at a time, under a lock file (`/tmp/uncrashed-agent.lock`) that also records pid, argv and, for write commands, a digest of the bound arguments. A waiting process prints `waiting for agent lock held by pid N (uc ...)` after about 2 s and gives up after `UNCRASHED_LOCK_TIMEOUT` with `agent busy: lock held by ... nothing was sent; retry when that process has finished` (the lock frees itself when its holder exits: never delete the file). A write command that acts (not a preview) whose digest equals the holder's fails at once with `the same command is already running as pid N; nothing was sent: wait for it instead of re-sending`. Waiting processes poll, so a holder that re-enters back to back can starve them; that fails loudly and sends nothing. With several agents, give `capture_view` an explicit pose and let one agent own the editor camera.

## Signals

SIGINT, SIGTERM and SIGHUP exit with 130, 143 and 129. Before anything was sent (waiting for the lock or an earlier job, or before the first command) the process exits at once with `error: interrupted while waiting; nothing was sent` or `interrupted during <op>; nothing was sent`; offline operations say `interrupted during <op>` only. Once a write command has sent a command, the first signal is deferred: the operation finishes, its result is printed, then the process exits with the signal's code, and `live state may be partial: ...` appears only then. A second signal aborts; one that arrives while the result is printed ends the output (`the operation finished, its output is incomplete`).

## Environment variables

| variable | effect |
| --- | --- |
| `UNCRASHED_OFFLINE` | any non-empty value: live agent calls fail at once (`offline mode ...`), nothing is injected. Offline-only commands keep working. |
| `UNCRASHED_FORMAT` | `json` makes `--json` the default |
| `UNCRASHED_LOCK` | agent lock file path (default `/tmp/uncrashed-agent.lock`) |
| `UNCRASHED_LOCK_TIMEOUT` | seconds to wait for the lock (default 60) |
| `UNCRASHED_DEBUG` | `1` prints the traceback of an unexpected error |
| `UNCRASHED_MAPS_DIR` | maps folder (default: `MapEditor/User` under the most recently used Steam profile in the standard Proton / Steam locations) |
| `UNCRASHED_STEAM_ID` | creator id (default: most recently used profile) |
| `UNCRASHED_WINE` | wine used to inject the DLL (default: Proton - Experimental's `files/bin/wine`) |
| `UNCRASHED_WINEPREFIX` | its prefix (default: `compatdata/1682970/pfx`) |
| `UNCRASHED_AGENT_DLL` | use this agent DLL instead of `native/uncrashed_agent.dll` |

## From MCP tools to CLI

Same operations, parameters and defaults. This table is generated from the registry in `api.py`; `./uc help <command>` shows the full parameter list with examples and the docstring. Uppercase words are positionals, `--name type=default` options, `--no-x` a boolean that defaults to true. "changes with --apply" previews until `--apply`; "changes at once" has no preview and no undo.

| MCP tool | usage | needs |
| --- | --- | --- |
| `status` | `./uc status` | nothing |
| `list_gates` | `./uc list_gates` | nothing |
| `search_assets` | `./uc search_assets [QUERY:str] [--category str] [--tag str] [--limit int=20] [--no-include_hidden] [--collision str]` | nothing |
| `get_assets` | `./uc get_assets IDS:str ...` | nothing |
| `list_maps` | `./uc list_maps` | nothing |
| `inspect_map` | `./uc inspect_map NAME:str` | nothing |
| `camera_pose` | `./uc camera_pose` | game |
| `move_camera` | `./uc move_camera X:num Y:num Z:num YAW:num [PITCH:num=-20] [--no-hold]` | game |
| `release_camera` | `./uc release_camera` | game |
| `capture_view` | `./uc capture_view [X:num] [Y:num] [Z:num] [YAW:num] [PITCH:num] [--backend str=engine] [--foreground] [--width int=800] [--height int=450] [--out str]` | game |
| `load_live` | `./uc load_live NAME:str [--no-move_camera]` | game; changes at once |
| `save_live` | `./uc save_live [NAME:str] [--display_name str]` | game; changes at once |
| `agent_exec` | `./uc agent_exec [--max_lines int=40] [--timeout num=30] COMMAND...` | game; changes at once |
| `agent_batch` | `./uc agent_batch <FILE \| - \| --commands CMD> [--no-quiet] [--max_lines int=40] [--timeout num]` | game; changes at once |
| `check_collisions` | `./uc check_collisions [--region x0,y0,x1,y1] [--top int=15] [--include_minor]` | game |
| `get_object_bounds` | `./uc get_object_bounds ACTOR:str` | game |
| `get_world_bounds` | `./uc get_world_bounds [SCOPE:str=placed]` | game |
| `check_foundations` | `./uc check_foundations [--actors a,b,..] [--region x0,y0,x1,y1] [--spacing num=0.75] [--tolerance num=0.3] [--max_drop num=50] [--include_structural] [--top int=15]` | game |
| `check_environment` | `./uc check_environment [--actors a,b,..] [--region x0,y0,x1,y1] [--spacing num=0.75] [--tolerance num=0.1] [--no-include_canopies] [--include_structural] [--top int=15]` | game |
| `sample_terrain` | `./uc sample_terrain POINTS:x,y[,z] ... [--z_from num=1000] [--z_to num=-2000]` | game |
| `agent_log` | `./uc agent_log [LINES:int=80]` | game |
| `agent_unload` | `./uc agent_unload` | game; changes at once |
| `inspect_site` | `./uc inspect_site BOUNDS_MIN:n,n,.. BOUNDS_MAX:n,n,.. [--spacing num=5] [--min_foliage_height num=2] [--top int=30]` | game |
| `clearing_zones` | `./uc clearing_zones [ACTION:str=list] [NAME:str] [--center n,n,..] [--size n,n,..] [--yaw num=0] [--apply] [--allow_partial]` | game; changes with --apply |
| `list_groups` | `./uc list_groups [PREFIX:str] [--limit int=50]` | game |
| `edit_group` | `./uc edit_group [GROUP:str] [--action str=transform] [--delta n,n,..] [--rotation n,n,..] [--scale num=1] [--pivot n,n,..] [--target_group str] [--prefix] [--actors a,b,..] [--apply]` | game; changes with --apply |
| `undo_edit` | `./uc undo_edit TRANSACTION:str [--apply]` | game; changes with --apply |
| `edit_history` | `./uc edit_history [--limit int=20]` | game |
| `define_assembly` | `./uc define_assembly [NAME:str] [--roles k=v,..] [--apply]` | game; changes with --apply |
| `check_assembly` | `./uc check_assembly NAME:str [--tolerance num=0.05] [--foundation_tolerance num=0.3] [--min_anchor_coverage num=0.5] [--top int=20]` | game |
| `set_collision` | `./uc set_collision [MODE:str=flythrough] [--actors a,b,..] [--group str] [--row str] [--box_min n,n,..] [--box_max n,n,..] [--channels a,b,..] [--keep_body] [--force] [--apply] [--no-persist] [--top int=30]` | game; changes with --apply |
| `collision_state` | `./uc collision_state [--actors a,b,..] [--group str] [--row str] [--box_min n,n,..] [--box_max n,n,..] [--channels a,b,..] [--top int=30] [--no-components]` | game |
| `undo_collision` | `./uc undo_collision TRANSACTION:str [--apply]` | game; changes with --apply |
| `noncolliding_alternatives` | `./uc noncolliding_alternatives ROW:str [--limit int=5]` | nothing |
| `check_performance` | `./uc check_performance [NAME:str] [--region x0,y0,x1,y1] [--budgets k=v,..\|json] [--measure] [--poses json] [--duration num=1.5] [--cell num=50] [--top int=10]` | saved map name, else game |
| `consolidate_scenery` | `./uc consolidate_scenery [NAME:str] [--region x0,y0,x1,y1] [--rows a,b,..] [--apply] [--tolerance num=0.02] [--max_size num=64] [--limit int=20]` | saved map name, else game; changes with --apply |
| `check_route_clearance` | `./uc check_route_clearance [SOURCE:str=live] [--points x,y[,z]\|json] [--profile str=5inch] [--margin_m num] [--spacing_m num=1] [--dims_m n,n,..] [--kind str] [--channel str=Pawn] [--probe_m num=10] [--directions int=16] [--closed\|--no-closed] [--laps int] [--no-align_gates] [--ignore_start_m num=0] [--max_issues int=20]` | game |
| `capture_sheet` | `./uc capture_sheet POSES:str [--width int=480] [--height int=270] [--columns int=0] [--out str]` | game |
| `scene_snapshot` | `./uc scene_snapshot [--group str] [--box n,n,..] [--name str]` | game |
| `scene_diff` | `./uc scene_diff SNAPSHOT:str [--group str] [--box n,n,..] [--position_m num=0.01] [--rotation_deg num=0.1] [--scale num=0.001] [--gate_position_m num=0.01] [--examples int=10]` | game |
| `check_opening` | `./uc check_opening PATH:x,y[,z] ... [--profile str=5inch] [--margin_m num] [--channel str=Pawn] [--spacing_m num=0.25] [--dims_m n,n,..] [--near_m num=5] [--top int=10]` | game |

`mcp` tool names such as `agent_exec` still work as CLI command names. The MCP server (`./uc mcp`, `uncrashed-mcp`, `python -m uncrashed_mcp`) is a thin adapter over the same registry.

## Known limitations

- A native fault inside a quiet batch leaves the DLL's quiet flag set (`native/agent.c` `run_job` does not reset `quiet_batch`). Replies then lose their header and commands time out although they ran, until `./uc unload` re-injects the DLL. Pre-existing; a one-line native fix is pending.
- An interrupted `set_collision --apply` can leave a `collision_prepared` journal that `undo_collision` refuses.
- Nothing in the port was exercised against the running game: live behaviour is covered by mocks and by the unchanged native protocol. The per-call batch path is unverified against the real DLL.
- In a small reading test a weaker model (Haiku) made slightly more lookup mistakes on the compact text than on JSON. For exact figures use `--json` with `--pick`, or `jq`.
