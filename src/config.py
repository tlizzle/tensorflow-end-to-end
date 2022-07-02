import os

host = os.getenv("host", "localhost")

package_dir = os.path.dirname(os.path.abspath('__file__'))
# models_dir = os.path.join(package_dir, "models")
models_dir = "/opt/tensorflow-end-to-end"

categorical_features = [
    'street',
    'city', 
    'statezip', 
    'country', 
    'weekday'
]

continuous_features = [
    'price',
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'sqft_lot',
    'floors',
    'waterfront',
    'view',
    'condition',
    'sqft_above',
    'sqft_basement',
    'yr_built',
    'yr_renovated',
]

label_name = 'price'


trial_number = 3