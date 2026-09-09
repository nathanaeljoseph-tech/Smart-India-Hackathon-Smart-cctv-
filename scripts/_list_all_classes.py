import zipfile
import collections

z = zipfile.ZipFile('archive.zip')

classes = ['Fence_Climbing', 'Fighting', 'Robbery', 'Shooting', 'Stealing']

for cls in classes:
    seqs = collections.defaultdict(lambda: {'count': 0, 'split': '', 'samples': []})
    for n in z.namelist():
        if f'/{cls}/' in n and not n.endswith('/'):
            parts = n.split('/')
            split = parts[1]
            fname = parts[3]
            s = fname.split('_frame_')[0] if '_frame_' in fname else fname.rsplit('_', 1)[0]
            seqs[s]['count'] += 1
            seqs[s]['split'] = split
            if len(seqs[s]['samples']) < 3:
                seqs[s]['samples'].append(fname)
                
    print(f"\n==========================================")
    print(f"Class: {cls} | Total Video Sequences: {len(seqs)}")
    print(f"==========================================")
    sorted_seqs = sorted(seqs.items(), key=lambda x: x[1]['count'])
    for name, data in sorted_seqs:
        duration_sec = data['count'] / 30.0
        print(f"  [{data['split']}] {name:<26} | Frames: {data['count']:>4} (~{duration_sec:>5.1f}s) | Sample: {data['samples'][0]}")
