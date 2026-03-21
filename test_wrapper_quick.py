from src.neurobids_flow.moabb_wrapper import NBIDSFDataset

dataset = NBIDSFDataset(bids_root='./bids_output', task='workload')
print('Subjects:', dataset.subject_list)
print('Sessions:', dataset._sessions)
print('Events:', list(dataset.event_id.keys()))

data = dataset._get_single_subject_data(1)
for sess, runs in data.items():
    for run, raw in runs.items():
        print(f'{sess}/{run} | ch={len(raw.ch_names)} | dur={raw.times[-1]:.1f}s | annotations={len(raw.annotations)}')
