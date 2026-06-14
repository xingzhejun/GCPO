def set_chunk(samples, args):
    if args.new_fix_chunk:
        num_chunks = len(args.new_chunk_list)
        return num_chunks, 0, args.new_chunk_list
    num_chunks = samples["timesteps"].shape[1] // args.chunk_size
    last_chunk_size = samples["timesteps"].shape[1] % args.chunk_size
    chunk_sizes = [args.chunk_size for _ in range(num_chunks)]
    if last_chunk_size != 0:
        chunk_sizes.append(last_chunk_size)
        num_chunks = num_chunks + 1
    
    return num_chunks, last_chunk_size, chunk_sizes

def get_chunk_list(samples, batch_size):
    keys_to_zip = list(samples.keys())
    values_to_zip = []
    for i in range(batch_size):
        sample_i_data = []
        for key in keys_to_zip:
            value = samples[key]
            if isinstance(value, tuple):
                # if tuple
                sample_i_data.append(tuple(chunk[i] for chunk in value))
            else:
                # if tensor
                sample_i_data.append(value[i])
        values_to_zip.append(sample_i_data)
        
    samples_batched_list = [dict(zip(keys_to_zip, x)) for x in values_to_zip]
    return samples_batched_list