import zipfile
import collections

z = zipfile.ZipFile('archive.zip')
for split in ['Test', 'Train']:
    print(f"\n=================== Split: {split} ===================")
    classes = ['Fence_Climbing', 'Fighting', 'Robbery', 'Shooting', 'Stealing', 'Normal_Videos']
    for cls in classes:
        prefix = f'UCF-Crime with Fence Climbing/{split}/{cls}/'
        files = [n[len(prefix):] for n in z.namelist() if n.startswith(prefix) and not n.endswith('/')]
        seqs = collections.defaultdict(list)
        for f in files:
            if '_frame_' in f:
                seq_name = f.split('_frame_')[0]
            elif '_' in f:
                seq_name = f.rsplit('_', 1)[0]
            else:
                seq_name = f
            seqs[seq_name].append(f)
        print(f"\nClass: {cls} (Total frames: {len(files)}, Sequences: {len(seqs)})")
        for seq_name, f_list in sorted(seqs.items()):
            print(f"  - Sequence: '{seq_name}' ({len(f_list)} frames)")
