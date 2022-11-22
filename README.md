# fruit-gan

## Fruit-gan is a Label-CGAN trained to generate 5 type of fruits/vegetables: 

- 0.Apple_red
- 1.avocado
- 2.grape
- 3.onion
- 4.pinepple

Usage : main.py --cuda [bool] --train [bool] --data_dir [directory_of_dataset] --out_dir [directory_of_output] --epoch [int] --batch_size [int] --lr [learning_rate] --latent_dim [dimension_random_Z_vector] --classes [nr_classes_to_generate] --channels [nr_rgb_channel] --log_inteval [to_sample_training_progresses]  --seed [random_seed]
  
If no dataset is specified fruit-gan uses the custom dataloader (custom_dataloader.py) to train on dataset\360-fruit_dataset\fruits-360\divided_training.
  
Already trained model with mnist, custom dataloader for 4 fruits and custom dataloader for 5 fruits is in *saved_models* folder.
To test already trained model execute inference.py and specify the generator weights' directory in *saved_models* .

## Treining examples
  
 
