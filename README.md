# 2dv50e
degree project at bachelor level

## How to run machine_learning project:
1) install *python (3.6.4), scikit-learn* and *django-admin* 
2) run `cd machine_learner/` 
3) run `python3 manage.py runserver`
4) App will be running on http://localhost:8000/

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