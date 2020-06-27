# ManeFrame

## Setup

Follow the [setup instructions](https://gist.github.com/josephsdavid/ff964492762e17e69dbf18e6c356db0d) from David J up to the requirements.
Then use requirements.sh to install the requirements.

```bash
$ bash requirements.sh
```

## Submitting Jobs

An example job submission script is shown in `example_job.sh`

## Other Commands

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
