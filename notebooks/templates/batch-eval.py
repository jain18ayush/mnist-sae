def process_sae(sae, dataset):
    neuron_map = {k: [] for k in range(sae.hidden_dim)} # create the matrix of neuron stuffs
    act_cache = []
    # add a hook to the right places... layer 3 or 4
    for data in dataset:
    model(data)
    for data in dataset:
    sae(data)
    # process through the act_cache where each row is the activaiotns of teh hidden layer for the input at index i
    # pick top 5 samples per column and save them to a dataframe with neuron_idx: j, sample_idx = [], sample_activations = [], depth=1
    # now in auto interp pipeline for each call just pull those 10 images and pass in the appropriate activations! and now you have a label.
def auto_interp_image(json, dataset):
# load in the images, and then make a call
for row in df:
    images = get_images(datset, indices)
    text = prompt + f"activations are {json[neuron][sample_activaitions]}"
    label = get_label(images, text)
    labels.append(label)
#store them in a text file too
with open('labels.txt', 'a'):
    f.write(label )
    df[label] = labels


def process-meta(df, sae):
    dataset = sae.weights
    print(sae.weights.shape)
    for data in dataset:
        model(sae)
    # same processing as above where we pick to p 4 samples per column and save them to df with neuron_idx: j, sample_idx = [], sample_acts = [], depth = depth

def auto_interp_text(label_df, process_df):
    # should be given the df with label column and a df with the activaitions as well as samples
    for row in process_df:
        labels = label_df[row[indices]]
        text = prompt + labels + process_df[activaitons]
        label = get_label(text)
        labels = ..

    with open('labels.txt', 'a'):
        f.write(label)

    df[label] = labels

