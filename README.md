# Incremental Rule Discovery in Response to Parameter Updates

### Overview
This project proposes an incremental rule discovery algorithm. The goal is to efficiently mine rules from a dataset, ensuring their support and confidence meet specified thresholds. This algorithm allows for dynamic adjustments to these thresholds without restarting the entire process, saving time and computational resources.

### Key Features
- Incremental Rule Discovery: Adapts to changes in support (σ) and confidence (δ) thresholds, adding or removing rules as necessary.
- Minimized Recomputations: Adjusts to parameter updates without restarting the discovery process.
- Scalable Algorithms: Utilizes parallel processing to improve runtime efficiency.
Entity Enhancing Rules: Supports data quality improvements through specialized rules.

### Datasets
You can get datasets from [this page]()

### Quick Start Guide
#### Prerequisites
Ensure the following software versions are installed:
 
- Ubuntu 22.04 (tested)
- Scala 2.12.10
- SBT 1.9.2
- Apache Spark 3.5.1

#### Steps
1. **Download Datasets**: Obtain the CSV datasets from the specified URL.

2. **Organize Files**: Place datasets in a directory, e.g., <PROJECT_ROOT>/datasets, and parameter input files in <PROJECT_ROOT>/in.
3. **Compile the project** with following command
```shell
sbt +clean +assembly
```
Make sure there is a file named plugins.sbt located in the <PROJECT_ROOT>/project directory, and that it contains the following content.
```sbt
addSbtPlugin( "com.eed3si9n" %% "sbt-assembly" % "2.2.0")
addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.10.1")
```


3. Run the Algorithm: Use the following shell command to execute the algorithm:


```shell
spark-submit \
--master local[*] \
--deploy-mode client \
--class org.dsl.SparkMain \
--name "IncREE" \
--conf spark.driver.memory=16g \
--conf spark.executor.memory=8g \
--conf spark.local.dir=<PROJECT_ROOT> \
--conf spark.executor.instances=20 \
<PROJECT_ROOT>/target/scala-2.12/HUME-assembly-0.2.0-CDF.jar \
-d datasets/adult.csv \
-p in/exp-test.csv \
-n 20 \
-o adult_output \
-m inc -l -c
```

Note: Key command options include:
- Replace `<PROJECT_ROOT>` with the path to the root directory of your project.
- `-d`: Specifies the dataset path.
- `-p`: Specifies the parameter input file path.
- `-o`: Specifies the output file path (created if it doesn't exist).



### Input Parameter File 

Input Parameter File Format
The input parameter file should be in CSV format, structured as follows:

```csv
Supp1,Supp2,Conf1,Conf2,Recall,Radius
1E-06,1E-05,0.95,0.85,1.0,3
1E-06,1E-05,0.95,0.85,0.7,3
```

- `Supp1/Supp2`: Old and new support parameters.
- `Conf1/Conf2`: Old and new confidence parameters.
- `Recall`: The recall guarantee for the incremental miner (applicable for certain parameter variations).
- `Radius`: Sampling radius for the procedure.

This guide should help you set up and run the incremental rule discovery algorithm efficiently. For more detailed information, please refer to the documentation provided within the project.


