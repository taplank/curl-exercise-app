This repo is me experimenting with pose detection. The cv_proj is just pose detection with RCNN, but app.py is an actual exercise app you can play around with. \
When configured, it will display the pose video of a person exercising on top of your camera, allowing you to follow it as closely as possible. \
It's a little weird at first since things are mirrored, but this is required so it's possible for you to copy their movements. \
To run this, you need to get your own Kaggle API from kaggle.com. It'll give you a file called kaggle.json. \
Download archive.zip from https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video. Yes, its a lot -- 10 GB. I'll make another file that removes selected files from the dataset in a bit. \
Then unzip it to get the archive folder, and put app.py, kaggle.json, and workout_index.json in that same folder, then run app.py:

The app is interactive with buttons. \
Use p to delete an exercise from the dataset, and l to skip to the next pose video of the current exercise. \
p is very useful here for cleaning up bad data and only having the exercises you want. Feel free to be liberal -- you can always redownload the file. \
Use m to move to the next exercise, and n to go back to the previous one. \
The pose of the person doing the workout will follow you, so change your camera angle so it lines up nicely. \
You will get a "Yay!" if you are close enough, else "Keep going." \
Name of exercise is in top left. \ 
I will try to add more updates to the exercise app to make the UI cleaner, and ideally get the R-CNN fully working first. Though I'm happy of the work I put into it regardless. 

For cv_proj, you will need to install val2017 from the coco datasets using wget: \
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip 

start the RCNN like this: \
python cv_proj.py \                                                               
    --ann ~/datasets/coco/annotations/person_keypoints_val2017.json \
    --images ~/datasets/coco/val2017 \
    --epochs 10 \
    --batch 4 \
    --batches_per_epoch 100
    
Then run the RCNN in demo mode so you can see it:
python cv_proj.py --run_demo --resume_ckpt ./ckpt/model_final.pth --input_size 512




