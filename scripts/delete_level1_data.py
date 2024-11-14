from pathlib import Path
import datetime
import time
import shutil

operations_dir = Path("/dataz/dsa110/operations")
subdirs_to_clear = [
    ("correlator", "*.hdf5"),
    ("T1/beams", "*.dat"),
    ("T1", "*/*.fil"),
    ("T1", "22*"),
    ("T1", "23*"),
    ("T2/cluster_output", "*.cand"),
    ("T2/cluster_output", "*.json"),
    ("T3", "*.png"),
    ("T3", "*.json"),
]

cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=0.5)
print(
    f"Removing operation files last modified prior to "
    f"{cutoff.strftime('%Y-%m-%dT%H:%M:%S')} UTC")

for subdir, pattern in subdirs_to_clear:
    for file in (operations_dir / subdir).glob(pattern):
        modtime = datetime.datetime.fromtimestamp(file.stat().st_mtime)
        # modtime is timezone naive, so we set it to utc
        # lxc managed containers are all using utc
        modtime = modtime.replace(tzinfo=cutoff.tzinfo)
        if modtime < cutoff:
            print(f'Removing {file}')
            try:
                file.unlink()
            except:
                shutil.rmtree(file)
