from __future__ import print_function

from tensorflow.keras.layers import Conv2D, Input, Concatenate, Activation, Add, BatchNormalization, PReLU
from tensorflow.keras.activations import swish
from tensorflow.keras import Model



def res_block_gen(model, kernel_size, filters, strides, activation_func='regular', kernel_reg=None):
    gen = model

    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                   kernel_regularizer=kernel_reg)(model)
    model = BatchNormalization(momentum=0.5)(model)
    # Using Parametric ReLU
    if activation_func == 'regular':
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
            model)
    elif activation_func == 'swish':
        model = swish(model)
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                   kernel_regularizer=kernel_reg)(model)
    model = BatchNormalization(momentum=0.5)(model)

    model = Add()([gen, model])

    return model

def Demixer_ResNet(input_rows, input_cols, filter_size=3, num_filters=64, num_resblocks=16):
    num_views = 9
    gen_input = Input((input_rows, input_cols, num_views))

    model = Conv2D(filters=64, kernel_size=filter_size, strides=1, padding="same")(gen_input)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)

    gen_model = model

    # Using 16 Residual Blocks
    for index in range(num_resblocks):
        model = res_block_gen(model, filter_size, num_filters, 1, activation_func='regular')

    model = Conv2D(filters=64, kernel_size=filter_size, strides=1, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = Add()([gen_model, model])

    model = Conv2D(filters=num_views, kernel_size=filter_size, strides=1, padding="same")(model)
    model = Activation('sigmoid')(model)

    generator_model = Model(inputs=gen_input, outputs=model)

    return generator_model


def Reconstructor_ResNet(input_rows, input_cols, num_views, merge_mode='add', rf_depth=50, output_z=32, filter_size=3,
                         num_filters=64, num_resblocks_vs=16, num_resblocks_rf=16, kernel_reg=None):
    input_z = num_views + rf_depth
    inputs = Input((input_rows, input_cols, input_z))
    print("inputs shape:", inputs.shape)

    input_views = inputs[:, :, :, 0:num_views]
    input_rfv = inputs[:, :, :, num_views:]

    model = Conv2D(filters=num_filters, kernel_size=filter_size, strides=1, padding="same",
                   kernel_regularizer=kernel_reg)(input_views)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)

    gen_model = model

    for index in range(num_resblocks_vs):
        model = res_block_gen(model, filter_size, num_filters, 1, activation_func='regular', kernel_reg=kernel_reg)

    model = Conv2D(filters=num_filters, kernel_size=filter_size, strides=1, padding="same",
                   kernel_regularizer=kernel_reg)(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = Add()([gen_model, model])

    model = Conv2D(filters=output_z, kernel_size=filter_size, strides=1, padding="same", kernel_regularizer=kernel_reg)(
        model)
    model = Activation('sigmoid')(model)

    model_rf = Conv2D(filters=num_filters, kernel_size=filter_size, strides=1, padding="same",
                      kernel_regularizer=kernel_reg)(input_rfv)
    model_rf = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
        model_rf)

    gen_model_rf = model_rf

    for index in range(num_resblocks_rf):
        model_rf = res_block_gen(model_rf, filter_size, num_filters, 1, activation_func='regular',
                                 kernel_reg=kernel_reg)

    model_rf = Conv2D(filters=num_filters, kernel_size=filter_size, strides=1, padding="same",
                      kernel_regularizer=kernel_reg)(model_rf)
    model_rf = BatchNormalization(momentum=0.5)(model_rf)
    model_rf = Add()([gen_model_rf, model_rf])

    model_rf = Conv2D(filters=output_z, kernel_size=filter_size, strides=1, padding="same",
                      kernel_regularizer=kernel_reg)(model_rf)
    model_rf = Activation('sigmoid')(model_rf)

    if merge_mode == 'add':
        output_volume = Add()([model, model_rf])
    elif merge_mode == 'concat':
        output_volume = Concatenate(axis=-1)([model, model_rf])

    output_volume = Conv2D(filters=output_z, kernel_size=filter_size, strides=1, padding="same",
                           kernel_regularizer=kernel_reg)(output_volume)
    output_volume = Activation('sigmoid')(output_volume)

    reconstructor = Model(inputs=inputs, outputs=output_volume)
    return reconstructor

