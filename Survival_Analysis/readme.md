# Survival Analysis Homework 3

This repository contains the code and report for Homework 3: Survival Analysis.

The goal of the assignment is to model customer churn using Accelerated Failure Time (AFT) models, compare different parametric survival distributions, select a final model, calculate customer-level CLV, and identify valuable and at-risk customer segments.

## Files

- `Report.Rmd` — main R Markdown file with code, analysis, plots, and interpretation.
- `Report.pdf` — knitted report output.
- `telco.csv` — customer churn dataset.
- `requirements.txt` — package/setup notes, including manual installation instructions if needed.

## Methods

The analysis uses the `telco.csv` dataset. Customer tenure is treated as the survival time, and churn is encoded as the event indicator:

- `churn = Yes` → event occurred
- `churn = No` → censored observation

AFT models are fitted using all available `survreg` distributions covered in the lecture:

- extreme
- logistic
- gaussian
- weibull
- exponential
- rayleigh
- loggaussian
- lognormal
- loglogistic
- t

Models are compared using log-likelihood and visual inspection of survival curves. The Kaplan-Meier curve is used as the empirical benchmark.

The final model is used to calculate customer-level CLV using discounted survival probabilities, following the logic from the lecture slides.

## Main Outputs

The report includes:

- AFT model comparison table
- Kaplan-Meier and AFT survival curves in one plot
- Final reduced AFT model
- Coefficient interpretation using time ratios
- Customer-level CLV calculation
- CLV analysis by customer segments
- At-risk customer identification
- Estimated annual retention budget
- Retention recommendations

## How to Run

Open `Report.Rmd` in RStudio and knit the file.

Required R packages:

```r
library(tidyverse)
library(survival)
library(survminer)
library(survMisc)

Follow instructions on downloading survMisc in the report if necessity to run the code rises