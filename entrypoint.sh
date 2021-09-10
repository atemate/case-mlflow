#!/bin/env bash
## Loads environment variables from a file and execute passed arguments
## see more options: https://gist.github.com/atemate/c84ddb3361fe977dae51bdf91a9b4883

ENV_PATH=${ENV_PATH:-"./dotenv"}
export $(grep -v '^#' ${ENV_PATH} | xargs -d '\n')


## Execute passed arguments
eval "$@"
