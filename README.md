# 2dv50e
degree project at bachelor level

## How to run machine_learning project:
1) install *python (3.6.4), scikit-learn* and *django-admin*
2) unzip all the files in `machine_learning/training_data`  
3) run `cd machine_learning/` 
4) run `python3 manage.py runserver`
5) App will be running on http://localhost:8000/

## API
## Online Learning
-------------------

**Train classification model**
----
* **URL** `http://localhost:8000?type=classification&mode=training`
* **Method:** `POST`
* **Headers:** `Content-Type: application/json`
* **Body:**
        
        {
            "adaptations": {
                ....
            },
            "environment: {
                ....
            }
        }
* **Success Response:**
    * **Content:**  <br />
                
            { 
                "message": "trained successfully"
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
            "adaptations": {
                ....
            },
            "environment: {
                ....
            }
        }
* **Success Response:**
    * **Content:**  <br />
                
            { 
                "result": [
                    ....
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
            "adaptations": {
                ....
            },
            "environment: {
                ....
            }
        }
* **Success Response:**
    * **Content:**  <br />
                
            { 
                "message": "trained successfully"
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
            "adaptations": {
                ....
            },
            "environment: {
                ....
            }
        }
* **Success Response:**
    * **Content:**  <br />
                
            { 
                "result": [
                    ....
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