echo "variables encoding"
cd data_preparation
python variables_encoding.py

echo "missing values imputation"
python missing_values_imputation.py
cd ../classification
python train_test_split.py missing_values
python naive_bayes.py missing_values
python knn.py missing_values

echo "outliers treatment"
cd ../data_preparation
python outliers_treatment.py
cd ../classification
python train_test_split.py outliers
python naive_bayes.py outliers
python knn.py outliers

echo "scaling"
cd ../data_preparation
python scaling.py
cd ../classification
python train_test_split.py scaling
python naive_bayes.py scaling
python knn.py scaling

echo "balancing"
cd ../data_preparation
python balancing.py
cd ../classification
python naive_bayes.py balancing
python knn.py balancing
