# Common ManeFrame Commands

#### Start A Job

```bash
$ sbatch job_scrip.sh
```

#### See the Parition Queue

```bash
$ squeue -p partition_name
```

partition names are [here](http://faculty.smu.edu/csc/documentation/slurm.html)

#### Watch Your Jobs

```bash
$ watch -n 1 squeue -u username
```

-n: time in seconds

#### Cancel Jobs

Cancel by job name

```bash
$ scancel --name=\<job_name\>
```

Cancel by username
