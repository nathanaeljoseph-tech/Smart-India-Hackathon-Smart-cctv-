"""Inspect the Resources.zip archive (it uses deflate64, so try 7z/tar)."""
import subprocess
import collections

result = subprocess.run(
    ["tar", "-tf", "Resources.zip"],
    capture_output=True, text=True, cwd=r"C:\Users\Lenovo\SIH"
)

lines = (result.stdout + result.stderr).strip().splitlines()
print(f"Total entries: {len(lines)}")

# Collect top-level dirs
dirs = set()
exts = collections.Counter()
for n in lines:
    parts = n.replace("\\", "/").split("/")
    if parts:
        dirs.add(parts[0])
    if "." in n:
        ext = n.rsplit(".", 1)[-1].lower()
        exts[ext] += 1
    else:
        exts["(no_ext)"] += 1

print(f"\nTop-level dirs/files: {sorted(dirs)}")
print(f"\nAll extensions:")
for ext, count in exts.most_common():
    print(f"  .{ext}: {count}")

print(f"\nSample entries with 'video' or 'mp4' in the name:")
for n in lines:
    nl = n.lower()
    if "video" in nl or "mp4" in nl or "avi" in nl:
        print(f"  {n}")
