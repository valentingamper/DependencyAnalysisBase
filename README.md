# Repository Dependency Metrics Analysis

A Jupyter notebook to clone a GitHub repo, extract dependency metrics for Java (Maven & Gradle), Python, and JavaScript files across commits, and save the results as JSON.

---

## Table of Contents

1. [Setup](#setup)  
   1.1 [Clone Repository](#clone-repository)  
   1.2 [Imports](#imports)  
   1.3 [Global Variables](#global-variables)  
2. [Function Definitions](#function-definitions)  
   2.1 [Compute File Metrics](#compute-file-metrics)  
   2.2 [Dependency Processing](#dependency-processing)  
   2.3 [File Discovery](#file-discovery)  
   2.4 [Commit History & Version Parsing](#commit-history--version-parsing)  
   2.5 [Dependency Parsers](#dependency-parsers)  
   2.6 [Loading & Fetching Helpers](#loading--fetching-helpers)  
   2.7 [Utility Functions](#utility-functions)  
3. [Execution Workflow](#execution-workflow)  
   3.1 [Initialize Caches & Data Structures](#initialize-caches--data-structures)  
   3.2 [Clone or Update Repository](#clone-or-update-repository)  
   3.3 [Discover Target Files](#discover-target-files)  
   3.4 [List Commits](#list-commits)  
   3.5 [Parse Java POM Dependencies](#parse-java-pom-dependencies)  
   3.6 [Parse Python & JavaScript Dependencies](#parse-python--javascript-dependencies)  
   3.7 [Parse Java Gradle Dependencies](#parse-java-gradle-dependencies)  
   3.8 [Merge & Clean Results](#merge--clean-results)  
4. [Results](#results)  
5. [Appendix](#appendix)  

---
