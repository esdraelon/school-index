# School Index MCP — deploy runbook

**This directory is artifacts + the ordered procedure.** The
deploy is a human-checked step: work the checkpoints in order and
stop if one can't be met — container up → local answer → token
registered → gateway smoke → negative checks → siblings
undisturbed → timer armed.

## Artifacts

| File | Purpose |
|---|---|
| `Dockerfile` | python:3.12-slim image; non-root uid 10001; `python -m school_index.index_mcp` |
| `docker-compose.index.yml` | the `index-mcp` service (localhost:8766, corpus root + read-only models mounts) |
| `nginx-index.conf.snippet` | `location /mcp/index/` for haliffer.ululuth.com |
| `run_pipeline.py` | the pipeline chain (sync → extract → embed), token handled in-process |
| `school-index-pipeline.service` / `.timer` | one pipeline pass, every 6h at :45 |

## Decisions baked in (and why)

- **Data dir is shared, not copied.** The container mounts the
  host's `~/school-index-data` (the pipeline's corpus root) at
  `/data`. One corpus, one truth — the MCP reads what the pipeline
  wrote, minutes after it wrote it.
- **WAL is the coexistence contract.** The DB runs in WAL mode
  (the stage stores' factories set it) so the server's reads run
  beside the pipeline's writes. The data mount is therefore
  read-write (a WAL reader needs the `-wal`/`-shm` sidecars), while
  the *server* is read-only by construction: its store issues
  SELECTs only — no migrate, no schema creation — and a system
  test proves serving moves no rows.
- **Access without ownership transfer.** The data dir stays owned
  by `haliffer` (the pipeline user). The container user (uid
  10001) gets exactly the access it needs via POSIX ACLs:
  `setfacl -m u:10001:rwx` on the dir, `u:10001:rw` on corpus.db,
  and a default ACL so future sidecars inherit it. The models
  subtree is mounted read-only over itself.
- **One token, two checks** — the gateway registry (client
  `index-mcp`, paths `["/mcp/index"]`) and the app itself
  (`INDEX_MCP_TOKEN`), same as the school MCP.

## Ops notes

- **WAL sidecars and the two-uid rule (learned 2026-10-23).**
  `corpus.db`'s `-wal` / `-shm` sidecars are created by whichever
  process opens the DB — the MCP container (uid 10001) or the
  host pipeline (user `haliffer`) — and the creator's uid owns
  them. The data dir therefore carries **default ACLs** granting
  both uids rw on new files, and the existing sidecars carry
  matching ACLs. If the pipeline ever reports `attempt to write a
  readonly database`, check the **sidecar ACLs first**
  (`getfacl ~/school-index-data/corpus.db-wal`): corpus.db itself
  will look perfectly writable — it's the sidecar the other uid
  created that is blocking the write.

## Ordered deploy steps (on hal9k)

1. **Sync the code.** The hal9k clone (`~/school-index`) carries
   the milestone branch; the host venv (`.venv`) installs it
   editable (`pip install -e '.[dev]'`).
2. **Stage deploy-run.** Create `~/school-index/deploy-run/` with
   `docker-compose.index.yml` (copied from `deploy/`) and a `.env`
   (mode 600) containing a freshly generated `INDEX_MCP_TOKEN`.
   Never commit or print the token.
3. **Grant the data ACLs** (see above) on `~/school-index-data`.
4. **Build + start.** From deploy-run:
   `sudo docker compose -f docker-compose.index.yml up -d --build`;
   `sudo docker logs haliffer-index-mcp` shows the startup line
   (`backend=sqlite-vec model=onnx-bge-small ... listen=...`) with
   no census refusal.
   - **Checkpoint: container up.**
5. **Local answer.** Call the MCP on `http://127.0.0.1:8766/` with
   the index token (SDK client): `search_passages` answers.
   - **Checkpoint: local answer.**
6. **Register the token.** Back up
   `/var/www/haliffer/auth/tokens.json`
   (`tokens.json.bak-<date>`), then add the sha256 of the index
   token as client `index-mcp`, paths `["/mcp/index"]`. Caveat:
   the file is bind-mounted into the `haliffer-auth` container as
   a single file — an atomic-rename replace swaps the inode and
   the container keeps reading the old one; edit in place, or
   restart `haliffer-auth` after replacing.
   - **Checkpoint: token registered.**
7. **Merge the nginx snippet.** Add the `location /mcp/index/`
   block to `/etc/nginx/sites-enabled/haliffer.ululuth.com.conf`
   (back the file up outside sites-enabled first — no `.bak` files
   may remain inside sites-enabled), `nginx -t`, then
   `systemctl reload nginx`.
8. **Gateway smoke.** Through `https://haliffer.ululuth.com/mcp/index/`
   with the index token alone as Bearer: all three tools answer;
   `search_passages("Pythagorean theorem")` returns the Model
   Pythagorean Theorem passage in the top 3.
   - **Checkpoint: gateway smoke.**
9. **Negative checks.** No token on `/mcp/index/` → 401; the
   school token there → 403; the index token on `/mcp/school/` →
   403.
   - **Checkpoint: negative checks.**
10. **Siblings undisturbed.** `/mcp/school/`, `/mcp/grades/`,
    `/mcp/todo/`, and the decoy `/` answer as before.
    - **Checkpoint: siblings undisturbed.**
11. **Arm the pipeline timer.** Copy
    `school-index-pipeline.service` / `.timer` to
    `/etc/systemd/system/`, `systemctl daemon-reload &&
    systemctl enable --now school-index-pipeline.timer`. Run the
    chain once by hand first
    (`CORPUS_ROOT=~/school-index-data .venv/bin/python
    deploy/run_pipeline.py` from the clone) — expect a near-no-op:
    agenda decks may legitimately change. Confirm with
    `systemctl list-timers`.
    - **Checkpoint: timer armed.**

## Rollback

Remove the nginx location + reload, `docker compose down` in
deploy-run, remove the tokens.json entry (in place), disable the
timer. The corpus DB and pipeline are untouched by all of it — the
server never wrote to them.
