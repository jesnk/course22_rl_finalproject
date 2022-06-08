# 1st Submission.
This code based on tensorflow 2.4.1

Overall required package listed in package_info.txt


## Guide to training

* Agent Training & Saving command

| python chain_train.py

| python lava_train.py

Trained model will saved in ./saved_models/{chain,lava}

## Guide to evaluate model

If you try to downloaded model, place 'saved_model' directory on root folder
We are not modified chain_test.py & lava_test.py. just execute below command.

* Testing code

| python chain_test.py

| python lava_test.py