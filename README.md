# 2dv50e
This repository contains the code of the approach that we mentioned in the thesis called: *Applying Machine Learning to Reduce the Adaptation Space in Self-Adaptive Systems: an exploratory work* (http://www.diva-portal.org/smash/record.jsf?pid=diva2:1240014)

- next part of this research -  http://www.diva-portal.org/smash/record.jsf?pid=diva2:1341195

This repository contains two main modules:

1. **machine_learner**: This contains all the code related to machine learning

2. **simulation**: This module contains two sub-modules:
    
    1. **activforms**: This contains all the code related to ActivFORMS approach. In addition, it also has some code that connects this module to **machine_learner** module through HTTP.
    
    2. **simulator**: This contains all the code related to DeltaIOT simulator.

    Other than above two sub-modules, **simulation** module also contains some some folders/files (SMCConfig.properties, models, uppaal-verifyta). These folder/files are used by **activform** module during runtime.

## How to run:
-------------------------------
1. Download this repository

2. Install Docker (https://www.docker.com/) and start it on your machine

3. Open terminal and go in to the root directory (2dv50e-master) of this repository

4. In terminal, write `cd machine_learner/`

5. Now, we are in **machine_learner** module. Run this module by executing following commands in terminal:
    
    1. `docker-compose build`
    
    2. `docker-compose up`

    **NOTE**: You can verify this by going to http://localhost:8000 on browser. It will say *{'mesage': 'only POST requests are allowed'}*. `ctr + c` can be used to stop this module. If you want to run again, only use `docker-compose up` command. Before re-running, **please delete the old learning models** from following folders:
     
     - `machine_learner/machine_learner/trained_models/classification`

     - `machine_learner/machine_learner/trained_models/regression`

     Any change in code will be executed directly due to hot-reloading (no need to re-run every time, however must delete old learning models)

6. Open new terminal window and go in to the root directory (2dv50e-master) of this repository

7. In terminal, write `cd simulation/`

8. Now, we are in **simulation** module. Run this module (sub-modules will be automatically executed) by executing following commands in terminal:
    
    1. `docker-compose build`
    
    2. `docker-compose up`

    - This will start the feeback loop that will print out some data in each run. This data will be saved in `simulation/activforms/log/log.txt` file (you can view it live). Below is the print format of this data:
        
        - When **Mode = Testing, TaskType = Classification or Regrssion**:
            
            - While *# of adaptation cycles <=  \# of training cycles*:
                
                - adaptation cycle;start time;training time; end time
            
            - While *# of adaptation cycles > \# of training cycles*:  
                
                - adaptation cycle;start time;testing time; adaptation space;training time; end time
        
        - When **Mode = Comparsion, TaskType does not matter**:
            
            - While *# of adaptation cycles <=  \# of training cycles*:
                
                - adaptation cycle;start time;classification training time;regression training time; activform adaptation space;end time
            
            - While *# of adaptation cycles > \# of training cycles*:  
                - adaptation cycle;start time;classification prediction time; regression prediction time;classification adaptation space;regression adaptation space;activform adaptation space;
                classificaton training time;regression training time;saving data time;end time
        
        - When **TaskType = ActivFORMS, Mode does not matter**:
            
            - adaptation cycle;start time;adaptation space;end time

        - When all the adaptation cycles are executed:
            
            - packet loss;energy consumption
        
        
        **NOTE:** `ctr + c` can be used to stop this module. If you want to run again, first run **machine_learner** module (if not already running) than run this module by only using `docker-compose up` command.

## How to change settings:
----------------------------
1. Install VSCode (https://code.visualstudio.com/). We do not recommend any IDE (eclipse, intellij, etc.). VSCode has some plugins that we use in the development. They are not required to run the project. However, they can help you in the devlopment.
    
    - For docker: https://marketplace.visualstudio.com/items?itemName=PeterJausovec.vscode-docker
    
    - For java: https://marketplace.visualstudio.com/items?itemName=redhat.java
    
    - For python: https://marketplace.visualstudio.com/items?itemName=ms-python.python

    **Note**: The above plugins might ask you to download some more plugins. Please ignore them.

2. In `simulation/activforms/src/smc/SMCConnector.java`, we can change following settings:
    
    - \# of training cycles
    
    - TaskType:
        
        - Classification
        
        - Regression
        
        - ActivFORMS
    
    - Mode: 
        
        - Testing: Train the selected *TaskType* on *\# of training cycles*, and then start the testing
        
        - ActivFORMS: Execute ActivFORMS
        
        - Comparison: Execute *ActivFORMS* + *Classification* + *Regression* in parallel. Then train *Classification* and *Regression* on *\# of training cycles*, and then start the testing. On the other hand, ActivFORMS runs continously

3. In `simulation/activforms/src/mapek/FeedbackLoop.java`, inside the start(), we can change:
    
    - \# of adaptation cycles

    
    **NOTE**: You can change the settings in other modules too. However, to test our approach we only recommend to change the above settings. 

## API of online supervised learning
-------------------
**Train classification model**
----
* **URL** `http://localhost:8000?type=classification&mode=training`
* **Method:** `POST`
* **Headers:** `Content-Type: application/json`
* **Body:**
        
        {
            "features": [
                -2, 5, 2, ...
            ],
            "target": {
                0, 0, 1, ...
            }
        }
**NOTE**: features length should be 25 and they should be in order (17 SNR, 6 Distribution, and 2 Traffic). In target use 0 when the packet loss is >=10%, and 1 when it is <10\%

* **Success Response:**
    * **Content:**  <br />
                
            { 
                "message": "training successful"
            }

* **Error Responses:**
   * **Content:**  <br />
                
            { 
                "message": "training failed"
            }
        OR

            { 
                "message": "invalid request"
            }
---
**Test classification model**
----
* **URL** `http://localhost:8000?type=classification&mode=testing`
* **Method:** `POST`
* **Headers:** `Content-Type: application/json`
* **Body:**
        
        {
            "features": [
                -2, 5, 2, ...
            ]
        }
**NOTE**: features length should be 25 and they should be in order (17 SNR, 6 Distribution, and 2 Traffic)

* **Success Response:**
    * **Content:**  <br />
                
            { 
                "predictions": [
                    1, 1, 0, ....
                ],
                "adaptation_space": 100
            }

* **Error Responses:**
   * **Content:**  <br />
                
            { 
                "message": "testing failed"
            }
        OR

            { 
                "message": "invalid request"
            }
---
**Train regression model**
----
* **URL** `http://localhost:8000?type=regression&mode=training`
* **Method:** `POST`
* **Headers:** `Content-Type: application/json`
* **Body:**
        
        {
            "features": [
                -2, 5, 2, ...
            ],
            "target": {
                10.22, 3.3, 15.901, ...
            }
        }
**NOTE**: features length should be 25 and they should be in order (17 SNR, 6 Distribution, and 2 Traffic)

* **Success Response:**
    * **Content:**  <br />
                
            { 
                "message": "training successful"
            }

* **Error Responses:**
   * **Content:**  <br />
                
            { 
                "message": "training failed"
            }
        OR

            { 
                "message": "invalid request"
            }

---
**Test regression model**
----
* **URL** `http://localhost:8000?type=regression&mode=testing`
* **Method:** `POST`
* **Headers:** `Content-Type: application/json`
* **Body:**
        
        {
            "features": [
                -2, 5, 2, ...
            ]
        }
**NOTE**: features length should be 25 and they should be in order (17 SNR, 6 Distribution, and 2 Traffic)

* **Success Response:**
    * **Content:**  <br />
                
            { 
                "predictions": [
                    10.2, 3.33, 8.09, ....
                ],
                "adaptation_space": 100
            }

* **Error Responses:**
   * **Content:**  <br />
                
            { 
                "message": "testing failed"
            }
        OR

            { 
                "message": "invalid request"
            }
---
