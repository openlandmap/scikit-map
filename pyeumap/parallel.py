'''
Parallelization helpers
'''

def ThreadGeneratorLazy(worker, args, max_workers, chunk):
    import concurrent.futures
    from itertools import islice

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        group = islice(args, chunk)
        futures = {executor.submit(worker, *arg) for arg in group}

        while futures:
            done, futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in done:
                yield future.result()

            group = islice(args, chunk)

            for arg in group:
                futures.add(executor.submit(worker,*arg))

def ProcessGeneratorLazy(worker, args, max_workers, chunk):
    import concurrent.futures
    from itertools import islice

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        group = islice(args, chunk)
        futures = {executor.submit(worker, *arg) for arg in group}

        while futures:
            done, futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in done:
                yield future.result()

            group = islice(args, chunk)

            for arg in group:
                futures.add(executor.submit(worker,*arg))
