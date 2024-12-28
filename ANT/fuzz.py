import argparse
import json
from ANT.ant import generate_ant
from ANT.state_abstraction import sequence_ngram_fn, parent_state_ngram_fn, index_parent_state_ngram_fn
from generators.Random import RandomOracle
from generators.RL import RLOracle
from generators.GFN_trajectory_balance import GFNOracle_trajectory_balance
from generators.GFN_detailed_balance import GFNOracle_detailed_balance
from generators.GFN_local_search import GFNOracle_local_search
from generators.GFN_flow_matching import GFNOracle_flow_matching
from tqdm import tqdm


def fuzz(oracle, trials, unique_valid, valid, invalid, model, local_search_steps, verbose):

    valids = 0
    valid_set = set()
    invalid_set = set()

    progress_bar = tqdm(range(trials))
    for i in progress_bar:
        if verbose:
            tqdm.write("=========")

        tree, num_nodes, validity = generate_ant(oracle, MAX_DEPTH)

        if model == "LS":
            assert local_search_steps is not None

            for i in range(local_search_steps):
                oracle.choice_sequence = oracle.choice_sequence[:len(
                    oracle.choice_sequence)//2]
                depth = oracle.calculate_depth()
                new_tree, new_num_nodes, new_validity = generate_ant(
                    oracle, MAX_DEPTH, depth)

                if validity and tree.__repr__() not in valid_set:
                    tree, num_nodes, validity = new_tree, new_num_nodes, new_validity
                    break

        # print(tree.xml)
        if verbose:
            tqdm.write("Tree with {} nodes".format(num_nodes))

        if validity:
            if verbose:
                tqdm.write("\033[0;32m" + tree.__repr__() + "\033[0m")
            valids += 1
            if tree.__repr__() not in valid_set:
                valid_set.add(tree.xml)
                oracle.reward(unique_valid)
            else:
                oracle.reward(valid)
        else:
            if verbose:
                tqdm.write("\033[0;31m" + tree.__repr__() + "\033[0m")
            if tree.__repr__() not in invalid_set:
                invalid_set.add(tree.__repr__())
            oracle.reward(invalid)

        progress_bar.set_description("{} trials / \033[92m{} valids ({} unique)\033[0m / \033[0;31m{} invalids ({} invalids)\033[0m / {:.2f}% unique valids".format(
            i, valids, len(valid_set), i + 1 - valids, len(invalid_set), (len(valid_set)*100/valids if valids != 0 else 0)))

    print("--------Done--------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, dest="trials",
                        help="Number of trials", default=1000)
    parser.add_argument("--model", type=str, dest="model",
                        help="Experiment type. RL / FM / TB / DB / LS", default="TB")
    parser.add_argument("--depth", type=int, dest="depth",
                        help="Max depth of the tree", default=3)
    parser.add_argument("--value_range", type=int,
                        dest="value_range", help="Range of values", default=4)
    parser.add_argument("--state_abstraction", type=str, dest="state_abstraction",
                        help="State abstraction function", default="tree")
    parser.add_argument("--local_search_steps", type=int, dest="local_search_steps",
                        help="Number of local search steps", default=5)
    parser.add_argument("--epsilon", type=float,
                        dest="epsilon", help="Epsilon", default=0.25)
    parser.add_argument("--embedding_dim", type=int,
                        dest="embedding_dim", help="Embedding dimension", default=128)
    parser.add_argument("--hidden_dim", type=int,
                        dest="hidden_dim", help="Hidden dimension", default=128)
    parser.add_argument("--verbose", dest="verbose",
                        help="Verbose", action="store_true", default=False)

    args = parser.parse_args()

    # Training
    TRIALS = args.trials
    MODEL = args.model
    STATE_ABSTRACTION = args.state_abstraction
    LS_STEPS = args.local_search_steps
    EPSILON = args.epsilon
    EMBEDDING_DIM = args.embedding_dim
    HIDDEN_DIM = args.hidden_dim
    VERBOSE = args.verbose

    # POM
    MAX_DEPTH = args.depth

    TAGS, TEXTS, NUM_CHILDRENS = None, None, list(range(5))
    with open('ANT/config.json', 'r') as file:
        config = json.load(file)
        TAGS = config["tags"]

    DOMAINS = [(TAGS, 1), (NUM_CHILDRENS, 2)]

    # Rewards
    UNIQUE_VALID = 20 if MODEL == "RL" else 1
    VALID = 1
    INVALID = -1 if MODEL == "RL" else 10e-6

    # Fuzz args
    fuzz_kwargs = {
        "trials": TRIALS,
        "unique_valid": UNIQUE_VALID,
        "valid": VALID,
        "invalid": INVALID,
        "model": MODEL,
        "local_search_steps": LS_STEPS,
        "verbose": VERBOSE
    }

    if MODEL == "RL":
        print()
        print(f"\033[1m==========Running RL with {STATE_ABSTRACTION} state abstraction function\033[0m==========".format(
            STATE_ABSTRACTION))
        if STATE_ABSTRACTION == "random":
            fuzz_kwargs["oracle"] = RandomOracle(DOMAINS)

        elif STATE_ABSTRACTION in ["sequence", "tree", "index_tree"]:
            oracle_kwargs = {
                "domains": DOMAINS,
                "epsilon": EPSILON,
            }
            if STATE_ABSTRACTION == "sequence":
                oracle_kwargs["abstract_state_fn"] = sequence_ngram_fn(4)

            elif STATE_ABSTRACTION == "tree":
                oracle_kwargs["abstract_state_fn"] = parent_state_ngram_fn(
                    4, MAX_DEPTH)

            elif STATE_ABSTRACTION == "index_tree":
                oracle_kwargs["abstract_state_fn"] = index_parent_state_ngram_fn(
                    4, MAX_DEPTH)

            print(oracle_kwargs)
            fuzz_kwargs["oracle"] = RLOracle(**oracle_kwargs)

        else:
            print("Invalid state abstraction function")
            exit(1)

    elif MODEL == "FM":
        fuzz_kwargs["oracle"] = GFNOracle_flow_matching(
            EMBEDDING_DIM, HIDDEN_DIM, DOMAINS, epsilon=EPSILON)

    elif MODEL == "TB":
        fuzz_kwargs["oracle"] = GFNOracle_trajectory_balance(
            EMBEDDING_DIM, HIDDEN_DIM, DOMAINS, epsilon=EPSILON)

    elif MODEL == "DB":
        fuzz_kwargs["oracle"] = GFNOracle_detailed_balance(
            EMBEDDING_DIM, HIDDEN_DIM, DOMAINS, epsilon=EPSILON)

    elif MODEL == "LS":
        fuzz_kwargs["oracle"] = GFNOracle_local_search(
            EMBEDDING_DIM, HIDDEN_DIM, DOMAINS, epsilon=EPSILON)
        fuzz_kwargs["local_search_steps"] = LS_STEPS

    else:
        print("Invalid model")
        exit(1)

    fuzz(**fuzz_kwargs)
