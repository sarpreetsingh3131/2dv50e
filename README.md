# 2dv50e
Degree project at bachelor level. The report can be found at http://lnu.diva-portal.org/smash/record.jsf?pid=diva2%3A1240014&dswid=-7780

## How to run:
1. Install **Java (1.8), Python (3.6.4), Scikit-Learn, django-admin (2.0.2), VS Code,** and **Eclipse** on your system 
2. Clone or download the repository
3. Open **VS Code**. Click `View` -> `Extensions` and search `Python` and install `Python` extension
4. In **VS Code** click `File` -> `Open` and Open **machine_learner** project
5. On left side of **VS Code** you should see the folders and files present in **machine_learner** project. Click on `manage.py` file. Now, you must see **Python (3.6.4)** on the left bottom of **VS Code**. We already configured this project for Mac. However, if you cannot see it then configure this project according to the following instructions:
    - From the file hierarchy of this project click `.vscode` -> `settings.json`.
    - In `settings.json` file provide the path where you installed **Python (3.6.4)**
6. In **VS Code** open terminal by clicking `View` -> `Terminal`. Now enter `python3 manage.py runserver`. The **machine_learner** project should be running on http://localhost:8000/
7. Open **Eclipse**.  Click `File` -> `New` -> `Java Project`. Unclick `Use default location` and click on `Browse`. Open **Simulator** project
8. Right click on **Simulator** project. Then click `Build Path` -> `Configure Build Path` -> `Libraries` -> `Add External JARs`. Import all the JAR files from `Simulator` -> `lib` and click `Apply and close`
9. Similarly Open **MachineLearning_simulation** project in **Eclipse**. In addition, import all JAR files from its `lib` folder.
10. Import **Simulator** project in **MachineLearning_simulation** project by right clicking on `MachineLearning_simulation` -> `Build Path` -> `Configure Build Path` -> `Projects` -> `Add` -> `Simulator` -> `Ok` -> `Apply and close` 
11. Run this project from `MachineLearning_simulation/src/main/Main.java`. The print out is in the following format:
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

## How to change settings:
- In `MachineLearning_simulation/src/smc/SMCConnector.java`, we can configure following settings:
    - \# of training cycles
    - TaskType:
        - Classification
        - Regression
        - ActivFORMS
    - Mode: 
        - Testing: Train the selected *TaskType* on *\# of training cycles*, and then start the testing
        - ActivFORMS: Execute ActivFORMS
        - Comparison: Execute *ActivFORMS* + *Classification* + *Regression* in parallel. Then train *Classification* and *Regression* on *\# of training cycles*, and then start the testing. On the other hand, ActivFORMS runs continously
- In `MachineLearning_simulation/src/mapek/FeedbackLoop.java`, inside the start(), we can configure:
    - \# of adaptation cycles

## API
## Online Supervised Learning
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
