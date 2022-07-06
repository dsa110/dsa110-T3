from pathlib import Path
import datetime

operations_dir = Path("/dataz/dsa110/operations")
subdirs_to_clear = [
    ("correlator", "*.hdf5"),
    ("T1/beams", "*.dat"),
    ("T1", "*/*.cand"),
    ("T2/cluster_output", "*.cand")
    ("T2/cluster_output", "*.json")]

cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
print(
    f"Removing operation files last modified prior to "
    f"{cutoff.strftime('%Y-%m-%d')}")

for subdir, pattern in subdirs_to_clear:
    for file in (operations_dir / subdir).glob(pattern):
        if datetime.datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
            print(f'Removing {file}')
            # file.unlink()
