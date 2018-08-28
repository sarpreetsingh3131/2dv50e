# 2dv50e
degree project at bachelor level. The report can be found at http://lnu.diva-portal.org/smash/record.jsf?pid=diva2%3A1240014&dswid=-7780

## How to run:
1. Clone or download the repository
1. Install **Java (1.8), Python (3.6.4), Scikit-Learn** and **django-admin** 
2. Run `cd machine_learner/` 
3. Run `python3 manage.py runserver`
4. **machine_learner** will be running on http://localhost:8000/
5. Import **MachineLearning_simulation** and **Simulator** projetcs in Eclispse
6. Configure the projects by importing the libraries which are present in the `lib` folder
7. Open `MachineLearning_simulation/src/smc/SMCConnector.java`. Here we can configure following settings:
    - \# of training cycles
    - TaskType:
        - Classification
        - Regression
        - ActivFORMS
    - Mode: 
        - Testing: Train the selected *TaskType* on *\# of training cycles*, and then start the testing
        - ActivFORMS: Execute ActivFORMS
        - comparison. Execute *ActivFORMS* + *Classification* + *Regression*. Then train *Classification* and *Regression* on \# of training cycles*, and then start the testing. On the other hand, ActivFORMS run continously
8. Open `MachineLearning_simulation/src/mapek/FeedbackLoop.java`. In the start(), we can configure *# of adaptation cycles*
9. Run this project from `MachineLearning_simulation/src/main/Main.java`. The print out is in the following format:
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
