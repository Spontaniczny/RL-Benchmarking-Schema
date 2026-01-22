# RL-Benchmarking-Schema
You can run your own RL experiments and see them benchmarked!  

It's required to have:
- uv installed (both locally and on Athena) for our automatic virtual environment setup to work corerctly 
    You can install it using:
    ```python
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    Verify if it's avaialable: 
    ```python
    uv --version
    ```

- swift (locally only) for gymnasium to work 
    ```python
    sudo apt update
    sudo apt install swig
    ```

## If you're running it locally
Use this command in your terminal:

```python
python main.py <path to the JSON experiment file>
```

## If you want to set it up on Athena (Cyfronet)
We highly recommend:
- only keeping the **repo** in your `$HOME`
- the **venv and logs** are best kept in your `$SCRATCH`.
- Using Remote-SSH in VSCode

You can (should) **set the paths**:
- to your **logs** folder in `config.json` (models and TensorBoard logs)
- to the folder where you want the **.RL_venv** file to be installed in `config_cyfronet.json`

### How to use RL-Benchmarking-Schema:
Using `submit_jobs.py` you can: 

- **install** all necessary project **dependencies** into the folder you defined in `config_cyfronet.json`
    
    Use an interactive session:
    ```python
    srun -N 1 --ntasks-per-node=1 -t 1:00:00 -p plgrid-now -A <grantname> --pty /bin/bash
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

### TensorBoard visualisation on Athena
If you're connected to Athena using Remote-SSH in VSCode, then this should work:

- On **login** node, activate the `.Rl_venv`:
    ```python
    source /net/tscratch/people/<username>/<project_folder>/.RL_venv/bin/activate
    ```

- Then to setup TensorBoard, run:
    ```python
    tensorboard --logdir /net/tscratch/people/<username>/<project_folder>/logs/Initial_Benchmarking/tb --port 6006 --host 127.0.0.1
    ```

If that worked, you should be able to see your beautiful plots @ `http://127.0.0.1:6006/`