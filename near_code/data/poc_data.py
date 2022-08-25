import numpy as np

PER_PROGRAM_TRAIN_SIZE = 500
PER_PROGRAM_TEST_SIZE = 50

def generate_data(programs, range_lower_upper, SIZE, NAME):
    xs,ys = [],[]
    for prog in programs:
        x = np.random.uniform(range_lower_upper[0], range_lower_upper[1], size=(SIZE,1))
        y = prog(x)
        xs.append(x)
        ys.append(y)
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    # numpy dump xs and ys in npy files
    print(f"Storing {NAME} data in xs_{NAME} and ys_{NAME} with sizes {xs.shape} and {ys.shape} ; hope dataloader permuts them")
    np.save(f'xs_{NAME}.npy', xs)
    np.save(f'ys_{NAME}.npy', ys)
    return

if __name__ == '__main__':
    programs = [
        lambda x : np.where(x<1/3, x+1, x-1),
        # lambda x : x+1,
        #lambda x : np.where(x>0,x,-x),
    ]
    range_lower_upper = [-1, 3]
    generate_data(programs, range_lower_upper, PER_PROGRAM_TRAIN_SIZE, "train")
    generate_data(programs, range_lower_upper, PER_PROGRAM_TEST_SIZE, "test")