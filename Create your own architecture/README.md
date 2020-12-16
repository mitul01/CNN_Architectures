# How to create your own architecture?
-- Steps to follow:
        ---> Develop an idea of architecture to implement.
        ---> Import the cnn blocks from cnn_blocks.py file required for your architecture. 
             Example:- <br>
             ```
             --> from cnn_blocks import inception_residual_block_A,inception_residual_block_B,inception_residual_block_C <br>
             --> from cnn_blocks import inception_block_reduction_A,inception_block_reduction_B <br> 
             --> from cnn_blocks import residual_block_v2 <br> 
             ```
