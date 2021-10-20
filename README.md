This repository contains the files and data used in the study "Performance Evaluation of Aggregation-based Group Recommender Systems for Ephemeral Groups"

## Project Structure

- data (Datasets)
  - CAMRA2011
  - GGF (FOOD)
  - ML (MovieLens)
  - THI (TOOLS)

- GRS_evaluation (Models)
  - profiles_agg (PROF aggregation strategy)
    - logs (Placeholder for the resulting logs)
    - model (Placeholder for the auxiliary models)
  - ratings_agg (PRED aggregation strategy)
    - logs (Placeholder for the resulting logs)
    - model (Placeholder for the auxiliary models)
    
## Example of execution
python main_tests.py -gs s -ng 100 -gt r -agg r -af avg -rs svd -m hr -d op -dpath "../data/" -mpath "../GRS_Evaluation/ratings_agg/models"

### Arguments

-gs: Group Size, Values: {s, m, l, vl, all}

-g: Groups, Values: An integer representing the number of groups to use

-gt: Group Type, Values: {r, s, d, rl, all}

-agg: Aggregation Strategy, Values: {r, p}

-af: Aggregation Function, Values: {avg, add, app, awm, bc, lm, mp, mrp, mul, pop, all}

-rs: Recommender Sytem, Values: {ibcf, ubcf, iucf, cb, hybrid, scd, ncf, all}

-m: Metrics, Values: {hr, ndcg, diversity, coverage, all}

-d: Dataset, Values: {thi, ggf, ml, camra2011, all}

-dpath: Data path. The path to the dataset

-mpath: Model path. The path to the auxiliary model

