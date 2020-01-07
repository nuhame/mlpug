import os
import sys
import random
import time

import torch
import torch.nn as nn

from mlpug.reference.chatbot_tutorial.loss import cross_entropy, masked_loss
from mlpug.reference.chatbot_tutorial.model_data_generation import batch2TrainData

# NVIDIA Automatic Mixed Precision Module
# Only imported when needed
amp = None


class Seq2SeqTrainModel(nn.Module):
    """
    Use this module to make training fast when using nn.DataParallel()
    """
    def __init__(self, encoder, decoder):
        super(Seq2SeqTrainModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                chat_inputs,
                seq_lengths,
                init_decoder_input,
                max_output_len,
                expected_bot_outputs,
                use_teacher_forcing):
        batch_size = chat_inputs.size(1)

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(chat_inputs, seq_lengths)

        decoder_n_layers = self.decoder.n_layers
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder_n_layers]

        decoder_input = init_decoder_input

        # What happens to this, will this be available directly on the right device?
        per_sample_cross_entropy = torch.cuda.FloatTensor(max_output_len, batch_size)
        # Forward batch of sequences through decoder one time step at a time
        for t in range(max_output_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            if use_teacher_forcing:
                decoder_input = expected_bot_outputs[t]
                decoder_input = decoder_input.unsqueeze(0)
            else:
                # No teacher forcing: next input is decoder's own current output
                _, top_1_index = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[top_1_index[i][0] for i in range(batch_size)]])

            decoder_output = decoder_output.squeeze()
            per_sample_cross_entropy[t, :] = cross_entropy(decoder_output, expected_bot_outputs[t])

        return per_sample_cross_entropy


def train(input_variable, lengths, init_decoder_input, max_output_len, target_variable, mask, teacher_forcing_ratio,
          train_model, optimizer, use_mixed_precision, clip):

    # mixed precision
    global amp

    # Zero gradients
    optimizer.zero_grad()

    # Initialize variables

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    per_sample_loss = train_model(input_variable, lengths, init_decoder_input, max_output_len,
                                  target_variable, use_teacher_forcing)

    loss = masked_loss(per_sample_loss, mask)

    # Perform back propagation
    if use_mixed_precision:
        if not amp:
            from apex import amp

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(train_model.parameters(), clip)

    # Adjust model weights
    optimizer.step()

    return loss.item()


def trainIters(model_name, voc, pairs, PAD_token, EOS_token, SOS_token, train_model, optimizer,
               model_checkpoint_dir, n_iteration, batch_size, teacher_forcing_ratio, use_mixed_precision, clip,
               print_every, save_every, corpus_name, checkpoint, device):

    # Was DataParallel applied?
    if hasattr(train_model, 'encoder'):
        embedding = train_model.encoder.embedding
        encoder = train_model.encoder
        decoder = train_model.decoder
    else:
        module = train_model.module
        embedding = module.encoder.embedding
        encoder = module.encoder
        decoder = module.decoder

    embedding_dim = embedding.embedding_dim
    encoder_state_size = encoder.hidden_size
    encoder_n_layers = encoder.n_layers
    decoder_n_layers = decoder.n_layers

    # Load batches for each iteration
    print("Generating data batches ...")
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)], PAD_token, EOS_token)
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if checkpoint and 'iteration' in checkpoint:
        start_iteration = checkpoint['iteration'] + 1

    init_decoder_input = torch.tensor([[SOS_token for _ in range(batch_size)]], dtype=torch.long, device=device)

    # Training loop
    print("Training...")
    tot_iter_time = 0
    for iteration in range(start_iteration, n_iteration + 1):
        start_time = time.time()

        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        # Run a training iteration with batch
        loss = train(input_variable, lengths, init_decoder_input, max_target_len, target_variable, mask, teacher_forcing_ratio,
                     train_model, optimizer, use_mixed_precision, clip)
        print_loss += loss

        end_time = time.time()
        iter_time = end_time-start_time
        tot_iter_time += iter_time

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}; Average iter. duration: {}ms".format(
                iteration,
                iteration / n_iteration * 100,
                print_loss_avg,
                int(1000*tot_iter_time/print_every)))
            tot_iter_time = 0
            print_loss = 0

        # Save checkpoint
        if iteration % save_every == 0 or iteration == n_iteration:
            print("Saving checkpoint ... ")
            directory = os.path.join(model_checkpoint_dir, model_name, corpus_name, '{}-{}-{}-{}'.format(
                embedding_dim,
                encoder_n_layers,
                decoder_n_layers,
                encoder_state_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'opt': optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

        sys.stdout.flush()
