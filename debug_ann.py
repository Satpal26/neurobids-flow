from neurobids_flow.moabb_wrapper import NBIDSFDataset

dataset = NBIDSFDataset(bids_root='./bids_output', task='workload')
data = dataset._get_single_subject_data(1)
raw = data['session_01']['run_0']
print('Annotations:')
for ann in raw.annotations:
    print('  desc=' + ann['description'] + '  onset=' + str(round(ann['onset'], 2)))
