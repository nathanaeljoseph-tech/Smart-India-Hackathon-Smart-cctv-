import zipfile
import collections

z = zipfile.ZipFile('archive.zip')

for cls in ['Fence_Climbing', 'Fighting']:
    seqs = collections.defaultdict(lambda: {'count': 0, 'split': '', 'samples': []})
    for n in z.namelist():
        if f'/{cls}/' in n and not n.endswith('/'):
            parts = n.split('/')
            split = parts[1]
            fname = parts[3]
            s = fname.split('_frame_')[0] if '_frame_' in fname else fname.rsplit('_', 1)[0]
            seqs[s]['count'] += 1
            seqs[s]['split'] = split
            if len(seqs[s]['samples']) < 2:
                seqs[s]['samples'].append(fname)
    print(f"\n====================== {cls} ({len(seqs)} videos) ======================")
    for s, d in sorted(seqs.items(), key=lambda x: x[1]['count']):
        dur = d['count'] / 30.0
        print(f"[{d['split']}] {s:<24} | {d['count']:>4} frames (~{dur:>5.1f}s) | sample: {d['samples'][0]}")
