-   [Introduction](#introduction)
    -   [Required packages](#required-packages)
    -   [meuse dataset](#meuse-dataset)
    -   [split training (tr) and test (ts)
        set](#split-training-tr-and-test-ts-set)
    -   [setting generic variables](#setting-generic-variables)
    -   [setting generic accuracy plot
        variables](#setting-generic-accuracy-plot-variables)
    -   [Loading required libraries:](#loading-required-libraries)
-   [References](#references)

Follow me on [![alt
text](http://i.imgur.com/tXSoThF.png "twitter icon with padding")](https://twitter.com/sheykhmousa)


Introduction
------------
`eumap` aims at providing easier access to EU environmental maps.
Basic functions train a spatial prediction model using [mlr3 package](https://mlr3.mlr-org.com/), [@mlr3], and related extensions in the [mlr3 ecosystem](https://github.com/mlr-org/mlr3/wiki/Extension-Packages) [@casalicchio2017openml; @MichelLang2020mlr3book], 
which includes spatial prediction using [Ensemble Machine
Learning](https://koalaverse.github.io/machine-learning-in-R/stacking.html#stacking-software-in-r/)
taking spatial coordinates and spatial cross-validation into account. In
a nutshell one can `train` an arbitrary `s3` **(spatial)dataframe** in
`mlr3` ecosystem by defining *df* and *target.variable* i.e., response.
main functions are as the following:

1.  `train.spm()` 1.1 `train.spm()` will automatically perform
    `classification` or `regression` tasks and the output is a
    `train.model` which later can be used to predict `newdata`.It also
    provides *summary* of the model and *variable importance* and
    *response*. The rest of arguments can be either pass or default
    values will be passed. 1.2 `train.spm()` provides four scenarios:

    1.  `classification` task with **non spatial** resampling methods
    2.  `regression` task with **non spatial** resampling methods
    3.  `classification` task with **spatial** resampling methods
    4.  `regression` task with **spatial** resampling methods

2.  `predict.spm()` 2.1. Prediction on a new dataset using `train.model`
    2.2. User needs to set`df.ts = test set` and also pass the
    `train.model`.

3.  `accuracy.plot()` 3.1 Accuracy plot in case of regression task
    (don’t use it for classification tasks for obvious reason), 3.2 in
    case of regression task,

    -   for now we have two scenarios including:
        -   rng = “nat” provides visualizations with real values
        -   rng = “norm” provides visualizations with the normalized
            (0~1) values note: don’t use it for classification tasks for
            obvious reasons.

**Warning:** most of functions are optimized to run in parallel by
default. This might result in high RAM and CPU usage.

The following examples demostrates spatial prediction using the meuse
data set:

### Required packages

    start_time <- Sys.time()
    ls <- c("lattice", "raster", "plotKML", "ranger", "mlr3verse", "BBmisc", "knitr", "bbotk",
        "hexbin", "stringr", "magrittr", "sp", "ggplot2", "mlr3fselect", "mlr3spatiotempcv", 
        "FSelectorRcpp", "future", "future.apply", "mlr3filters", "EnvStats", "grid", "mltools","gridExtra","yardstick","plotKML", "latticeExtra","devtools")
    new.packages <- ls[!(ls %in% installed.packages()[,"Package"])]
    if(length(new.packages)) install.packages(new.packages, repos="https://cran.rstudio.com", force=TRUE)

### meuse dataset

    library("sp")
    demo(meuse, echo=FALSE)
    pr.vars = c("x","y","dist","ffreq","soil","lead")
    df <- as.data.frame(meuse)
    df.grid <- as.data.frame(meuse.grid)
    # df <- df[complete.cases(df[,pr.vars]),pr.vars]
    df = na.omit(df[,])
    df.grid = na.omit(df.grid[,])
    summary(is.na(df))

         x               y            cadmium          copper       
     Mode :logical   Mode :logical   Mode :logical   Mode :logical  
     FALSE:152       FALSE:152       FALSE:152       FALSE:152      
        lead            zinc            elev            dist        
     Mode :logical   Mode :logical   Mode :logical   Mode :logical  
     FALSE:152       FALSE:152       FALSE:152       FALSE:152      
         om            ffreq            soil            lime        
     Mode :logical   Mode :logical   Mode :logical   Mode :logical  
     FALSE:152       FALSE:152       FALSE:152       FALSE:152      
      landuse          dist.m       
     Mode :logical   Mode :logical  
     FALSE:152       FALSE:152      

    summary(is.na(df.grid))

       part.a          part.b           dist            soil        
     Mode :logical   Mode :logical   Mode :logical   Mode :logical  
     FALSE:3103      FALSE:3103      FALSE:3103      FALSE:3103     
       ffreq             x               y          
     Mode :logical   Mode :logical   Mode :logical  
     FALSE:3103      FALSE:3103      FALSE:3103     

    crs = "+init=epsg:28992"
    target.variable = "lead"

### split training (tr) and test (ts) set

    smp_size <- floor(0.5 * nrow(df))
    set.seed(123)
    train_ind <- sample(seq_len(nrow(df)), size = smp_size)
    df.tr <- df[, c("x","y","dist","ffreq","soil","lead")]
    df.ts <- df.grid[, c("x","y","dist","ffreq","soil")]

### setting generic variables

    folds = 2
    n_evals = 3
    newdata = df.ts

### setting generic accuracy plot variables

    colorcut. = c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1)
    colramp. = colorRampPalette(c("wheat2","red3"))
    xbins. = 50

### Loading required libraries:

    library("mlr3verse")
    library("bbotk")
    library("ggplot2")
    library("mltools")
    library("data.table")
    library("mlr3fselect")
    library("FSelectorRcpp")
    library("future")
    library("future.apply")
    library("magrittr")
    library("progress")
    library("mlr3spatiotempcv")
    library("sp")
    library("landmap")  
    library("dplyr")
    library("EnvStats")
    library("grid")
    library("hexbin")
    library("BBmisc")
    library("lattice")
    library("MASS")
    library("gridExtra")
    library("MLmetrics")
    library("yardstick")
    library("plotKML")
    library("latticeExtra")
    library("devtools")
    library("raster")

`train.spm` fits multiple models/learners depending on the `class()` of
the **target.variable** and for returns a `trained model`, **var.imp**,
**summary** of the model, and **response** variables. `trained model`
later can predict a new dataset.

`predict.spm()`

prediction on new dataset

accuracy plot

calling `train.spm`

    tr = train.spm(df.tr, target.variable = target.variable , folds = folds , n_evals = n_evals , crs)

    Regr Task   resampling method: non-spatialCV  ncores:  32 ...TRUE

    Using learners: method.list...TRUE

               Fitting a ensemble ML using 'mlr3::Taskregr'...TRUE

`train.spm` results:

    train.model= tr[[1]]
    var.imp = tr[[2]]
    var.imp

        dist        x        y    ffreq     soil 
    472304.4 248899.5 226325.8 164080.2 105424.1 

    summary = tr[[3]]
    summary

    Ranger result

    Call:
     ranger::ranger(dependent.variable.name = task$target_names, data = task$data(),      case.weights = task$weights$weight, importance = "impurity",      mtry = 2L, sample.fraction = 0.751642505638301, num.trees = 287L) 

    Type:                             Regression 
    Number of trees:                  287 
    Sample size:                      152 
    Number of independent variables:  5 
    Mtry:                             2 
    Target node size:                 5 
    Variable importance mode:         impurity 
    Splitrule:                        variance 
    OOB prediction error (MSE):       4565.397 
    R squared (OOB):                  0.631821 

    response = tr[[4]]
    response

      [1] 245.74701 198.56257 175.88260 152.64985 109.78419  89.92654 133.32049
      [8] 187.69835 152.62631  96.49871  90.27788 140.63443 230.66843 151.79353
     [15] 152.21687 244.51578 201.96843 189.52022 231.13580 196.41175 166.55726
     [22] 162.23817  78.37787  81.37886  82.09700  78.87819  79.48634  91.46878
     [29]  94.90916  78.12105  90.56540  95.71523  79.73448  80.96752  97.79402
     [36] 122.95409 226.21821 264.94612 245.52608 180.21900 138.12811 212.15112
     [43] 155.95637 113.39823 118.35871 123.89889 127.37492 155.75373 360.12410
     [50] 370.22072 358.34795 355.58192 342.90001 222.71403 251.20865 304.59883
     [57] 212.87052 262.12715 258.84917 195.99975 209.03627 200.06445 194.26583
     [64] 155.81713 234.73617 157.05111 274.29458 263.56792 260.68516 243.87849
     [71] 248.16529 246.23354 199.73629 209.53731 304.58005 269.35494 269.64654
     [78] 307.04231 281.91972 254.22778 126.47033 117.37150  93.88349 197.34422
     [85] 193.64448 155.66761 151.05174 116.01464 224.98667 231.01544  81.29955
     [92]  74.64146  75.86594  83.07239  92.53966  92.10977  80.13611  69.89051
     [99]  81.24273  72.64670  54.77423  60.35633  52.20705  52.68541  59.72386
    [106]  63.63051  57.43823  62.39574  59.22295  59.02173  74.37124 143.05840
    [113] 137.12194 108.81446 131.23716  64.76907  64.87401  90.07664 134.92331
    [120] 169.01150 227.58547 159.08946  73.86919  68.46705  67.18053 182.78634
    [127] 204.70729  50.46940  71.98755 110.08505  63.23185  58.47294  58.26195
    [134]  55.56485 113.58069 124.29915 118.72380  68.86217 108.32210 139.51111
    [141] 187.44688 107.48677 200.00631 163.21052 181.22554  99.22140 134.59688
    [148]  80.66763  87.35705 146.63598  78.32593 202.47290

calling `predict.spm()`

    predict.variable = predict.spm(train.model, newdata)

`predict.spm()` results:

    predict.variable

       [1] 260.96457 260.96457 233.66716 234.80209 260.96457 233.66716 231.43072
       [8] 234.44279 264.24775 233.45740 231.43072 234.44279 204.73008 193.59737
      .....
      .....
      .....
    [3053] 161.09919 184.88595 184.87006 178.38371 180.97604 197.40706 197.99460
    [3060] 320.99734 324.49647 303.47920 299.11811 304.17845 302.91831 285.57235
    [3067] 279.93406 188.35636 186.38011 175.82721 191.11774 190.11585 189.57880
    [3074] 186.05136 187.58092 189.01147 199.36873 196.77653 320.93462 319.76563
    [3081] 322.65774 315.35383 309.38298 306.18768 295.56417 296.56473 215.20839
    [3088] 205.47790 193.43609 194.85740 210.46077 208.74149 207.14581 204.24134
    [3095] 323.49008 325.73026 322.67222 320.85128 319.77166 307.73029 221.11597
    [3102] 211.57886 209.28414

calling `accuracy.plot.spm` … result

    plt = accuracy.plot.spm(x = df.tr[,target.variable], y = response, rng = "norm")

<img src="README_files/figure-markdown_strict/unnamed-chunk-14-1.png" alt="Accuracy plot"  />
<p class="caption">
Accuracy plot
</p>

make a raster grid out of predicted variables e.g., lead (in this case)

raster grid output:

    plot(df.ts[,"leadp"])
    points(meuse, pch="+")

<img src="README_files/figure-markdown_strict/unnamed-chunk-16-1.png" alt="Raster grid"  />
<p class="caption">
Raster grid
</p>

References
----------
