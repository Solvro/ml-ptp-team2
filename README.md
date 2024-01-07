# PTP1
Patch the Planet: Restore Missing Data | Team 2

## Challenge
https://thinkonward.com/app/c/challenges/patch-the-planet

## Data

- [Dataset](https://thinkonward.com/app/c/challenges/patch-the-planet/data)


## Installation steps

1. Create new virtual environment:
    
    ```
    conda create --name ptp python=3.10
    ```

2. Activate environment
    ```
    conda activate ptp
    ```

3. Update _pip_ version:
    ```
    python -m pip install --upgrade pip
    ```
4. Install _ptp_ package:

    ```
    python -m pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu121
    ```
5. Enable precommit hook:
    ```
    pre-commit install
    ```
