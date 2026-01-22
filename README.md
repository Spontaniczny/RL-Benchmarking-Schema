# RL-Benchmarking-Schema
RL Benchmarking Schema Project 

## If you're running it locally
Use this command in your terminal:

```python
python main.py <path to the JSON experiment file>
```

## If you want to set it up on Cyfronet
We highly recommend:
- only keeping the **repo** in your `$HOME`
- the **venv and logs** are best kept in your `$SCRATCH`.

You can (should) **set the paths**:
- to your **logs** folder in `config.json` (models and TensorBoard logs)
- to the folder where you want the **.RL_venv** file to be installed in `config_cyfronet.json`

### How to use RL-Benchmarking-Schema:
Using `submit_jobs.py` you can: 

- **install** all necessary project **dependencies** into the folder you defined in `config_cyfronet.json`
    
    Use an interactive session:
    ```python
    srun -N 1 --ntasks-per-node=1 -t 1:00:00 -p plgrid-now -A <grantname>-cpu --pty /bin/bash
    ```
    and then run:
    ```python
    python submit_jobs.py --setup-only
    ```

- **submit jobs** - each one is a separate experiment

    You must switch **back to the login node** (otherwise, throws an AccessDenied error).
    
    Then run:
    ```python
    python submit_jobs.py experiments/<path to the JSON experiment file>
    ```
