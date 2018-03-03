# 2dv50e
degree project at bachelor level

## How to run machine_learning project:
1) install *python (3.6.4), scikit-learn* and *django-admin*
2) run `cd machine_learning/` 
3) run `python3 manage.py runserver`
4) App will be running on http://localhost:8000/

## API
**Train classification model**
----
* **URL** `http://localhost:8000?type=classification&mode=training&model_name=sgd_classifier`
* **Method:** `GET`
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
* **URL** `http://localhost:8000?type=classification&mode=testing&model_name=sgd_classifier`
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
* **URL** `http://localhost:8000?type=regression&mode=training&model_name=sgd_regressor`
* **Method:** `GET`
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
* **URL** `http://localhost:8000?type=regression&mode=testing&model_name=sgd_regressor`
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
**Test accuracy of classification model**
----
* **URL** `http://localhost:8000?type=classfication&mode=accuracy&model_name=sgd_classifier`
* **Method:** `GET`
* **Success Response:**
    * **Content:**  <br />
                
            { 
                "accuracy": 0.95
            }

* **Error Responses:**
   * **Content:**  <br />
                
            { 
                "message": "accuracy failed"
            }
        OR

            { 
                "message": "invalid request"
            }

---

**Test accuracy of regression model**
----
* **URL** `http://localhost:8000?type=regression&mode=accuracy&model_name=sgd_regressor`
* **Method:** `GET`
* **Success Response:**
    * **Content:**  <br />
                
            { 
                "accuracy": 0.95
            }

* **Error Responses:**
   * **Content:**  <br />
                
            { 
                "message": "accuracy failed"
            }
        OR

            { 
                "message": "invalid request"
            }

---