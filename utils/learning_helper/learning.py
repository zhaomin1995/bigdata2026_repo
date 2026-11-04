import torch
from model_architecture.nn_no_matcher import NoMatcherModel
from model_architecture.nn_matcher1 import MatcherModel1
from model_architecture.neuralbaseline_text import NeuralBaselineText
from model_architecture.neuralbaseline_image import NeuralBaselineImage
from model_architecture.neuralbaseline_locdesc import NeuralBaselineLocDesc
from model_architecture.neuralbaseline_annotquestion import NeuralBaselineAnnotQuestion
from model_architecture.neuralbaseline_text_image import NeuralBaselineTextImage
from model_architecture.neuralbaseline_text_locdesc import NeuralBaselineTextLocDesc
from model_architecture.neuralbaseline_text_annotquestion import NeuralBaselineTextAnnotQuestion
from model_architecture.neuralbaseline_image_locdesc import NeuralBaselineImageLocDesc
from model_architecture.neuralbaseline_image_annotquestion import NeuralBaselineImageAnnotQuestion
from model_architecture.neuralbaseline_locdesc_annotquestion import NeuralBaselineLocDescAnnotQuestion
from model_architecture.nerualbaseline_text_locdesc_annotquestion import NeuralBaselineTextLocDescAnnotQuestion
from model_architecture.nerualbaseline_text_image_locdesc import NeuralBaselineTextImageLocDesc
from model_architecture.nerualbaseline_text_image_annotquestion import NeuralBaselineTextImageAnnotQuestion
from model_architecture.nerualbaseline_image_locdesc_annotquestion import NeuralBaselineImageLocDescAnnotQuestion
from model_architecture.nerualbaseline_text_image_locdesc_annotquestion import NeuralBaselineTextImageLocDescAnnotQuestion
from model_architecture.recursive_all1 import RecursiveAll1
from model_architecture.recursive_all2 import RecursiveAll2
from model_architecture.recursive_all3 import RecursiveAll3
from model_architecture.recursive_all4 import RecursiveAll4
from model_architecture.nb_simplify1_oneinput import NBSimplify1OneInput
from model_architecture.nb_simplify1_twoinputs import NBSimplify1TwoInputs
from model_architecture.nb_simplify1_threeinputs import NBSimplify1ThreeInputs
from model_architecture.nb_simplify1_fourinputs import NBSimplify1FourInputs
from model_architecture.nb_simplify2_fourinputs import NBSimplify2FourInputs
from model_architecture.nb_simplify2_threeinputs import NBSimplify2ThreeInputs
from model_architecture.nb_simplify2_twoinputs import NBSimplify2TwoInputs
from model_architecture.multiencoder1 import MultiEncoder1
from model_architecture.multiencoder2 import MultiEncoder2
from model_architecture.multiencoder3 import MultiEncoder3
from model_architecture.clip_based1 import CLIPEncoder1
from model_architecture.clip_based2 import CLIPEncoder2
from model_architecture.individual_encoders import IndividualEncoder
from model_architecture.shared_encoders import SharedEncoder
from model_architecture.text_only import TextOnly
from model_architecture.image_only import ImageOnly


class LearningHelper:

    def __init__(self):
        pass

    def get_model_name(self, model, use_image, use_loc_desc, use_user_desc, pooling, fine_tune):
        """

        :param model:
        :param use_image:
        :param use_loc_desc:
        :param use_user_desc:
        :param pooling:
        :param fine_tune:
        :return: the name of the model type -> string
        """
        types = [model.capitalize()]
        if use_image:
            types.append('Image')
        if use_loc_desc:
            types.append('LocDesc')
        if use_user_desc:
            types.append('UserDesc')
        if pooling:
            types.append(pooling)
        if fine_tune:
            types.append("FineTune")

        return '_'.join(types)

    def get_model(self, model, fine_tune, use_image, use_user_desc, use_loc_desc, pooling):
        """

        :param model:
        :param fine_tune:
        :param use_image:
        :param use_user_desc:
        :param use_loc_desc:
        :param pooling:
        :return:
        """
        # get the model based on matcher_type
        if model == 'no_matcher':
            classifier = NoMatcherModel(
                freeze_bart=not fine_tune,
                use_image=use_image,
                use_user_desc=use_user_desc,
                use_loc_desc=use_loc_desc,
                pooling=pooling
            )
        elif model == 'matcher1':
            classifier = MatcherModel1(
                freeze_bart=not fine_tune,
                use_image=use_image,
                use_user_desc=use_user_desc,
                use_loc_desc=use_loc_desc,
                pooling=pooling
            )
        elif model == 'neuralbaseline_text':
            classifier = NeuralBaselineText()
        elif model == 'neuralbaseline_image':
            classifier = NeuralBaselineImage()
        elif model == 'neuralbaseline_locdesc':
            classifier = NeuralBaselineLocDesc()
        elif model == 'neuralbaseline_annotquestion':
            classifier = NeuralBaselineAnnotQuestion()
        elif model == 'neuralbaseline_text_image':
            classifier = NeuralBaselineTextImage()
        elif model == 'neuralbaseline_text_locdesc':
            classifier = NeuralBaselineTextLocDesc()
        elif model == 'neuralbaseline_text_annotquestion':
            classifier = NeuralBaselineTextAnnotQuestion()
        elif model == 'neuralbaseline_image_locdesc':
            classifier = NeuralBaselineImageLocDesc()
        elif model == 'neuralbaseline_image_annotquestion':
            classifier = NeuralBaselineImageAnnotQuestion()
        elif model == 'neuralbaseline_locdesc_annotquestion':
            classifier = NeuralBaselineLocDescAnnotQuestion()
        elif model == 'neuralbaseline_text_locdesc_annotquestion':
            classifier = NeuralBaselineTextLocDescAnnotQuestion()
        elif model == 'neuralbaseline_text_image_locdesc':
            classifier = NeuralBaselineTextImageLocDesc()
        elif model == 'neuralbaseline_text_image_annotquestion':
            classifier = NeuralBaselineTextImageAnnotQuestion()
        elif model == 'neuralbaseline_image_locdesc_annotquestion':
            classifier = NeuralBaselineImageLocDescAnnotQuestion()
        elif model == 'neuralbaseline_text_image_locdesc_annotquestion':
            classifier = NeuralBaselineTextImageLocDescAnnotQuestion()
        elif model.startswith('recursiveall1_'):
            classifier = RecursiveAll1(model=model)
        elif model.startswith('recursiveall2_'):
            classifier = RecursiveAll2(model=model)
        elif model.startswith('recursiveall3_'):
            classifier = RecursiveAll3(model=model)
        elif model.startswith('recursiveall4_'):
            classifier = RecursiveAll4(model=model)
        elif model.startswith('nbsimplify1oneinput_'):
            classifier = NBSimplify1OneInput(model=model)
        elif model.startswith('nbsimplify1twoinputs_'):
            classifier = NBSimplify1TwoInputs(model=model)
        elif model.startswith('nbsimplify1threeinputs_'):
            classifier = NBSimplify1ThreeInputs(model=model)
        elif model.startswith('nbsimplify1fourinputs_'):
            classifier = NBSimplify1FourInputs(model=model)
        elif model.startswith('nbsimplify2fourinputs_'):
            classifier = NBSimplify2FourInputs(model=model)
        elif model.startswith('nbsimplify2threeinputs_'):
            classifier = NBSimplify2ThreeInputs(model=model)
        elif model.startswith('nbsimplify2twoinputs_'):
            classifier = NBSimplify2TwoInputs(model=model)
        elif model.startswith('multiencoder1_'):
            classifier = MultiEncoder1(model=model)
        elif model.startswith('multiencoder2_'):
            classifier = MultiEncoder2(model=model)
        elif model.startswith('multiencoder3_'):
            classifier = MultiEncoder3(model=model)
        elif model.startswith('clip1_'):
            classifier = CLIPEncoder1(model=model)
        elif model.startswith('clip2_'):
            classifier = CLIPEncoder2(model=model)
        elif model.startswith('individualencoders_'):
            classifier = IndividualEncoder(model=model)
        elif model.startswith('sharedencoders_'):
            classifier = SharedEncoder(model=model)
        elif model == 'text_only':
            classifier = TextOnly(model)
        elif model == 'image_only':
            classifier = ImageOnly(model)
        else:
            raise Exception("Please choose correct model type")
        return classifier

    def epoch_time(self, start_time, end_time):
        """Compute the time spent for each epoch.

        Args:
            start_time: the time before each epoch
            end_time: the time after each epoch
        Returns:
            the time spent for each epoch
        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train(self, model, iterator, optimizer, criterion, device, label_to_idx):
        """Train the model.

        Args:
            model: the model to be trained
            iterator: training data loader
            optimizer: optimizer
            criterion: loss function
            device: cpu or cuda
            label_to_idx: convert the label into index to use the CrossEntropy
        Returns:
            the loss and accuracy of each epoch
        """

        epoch_loss = 0
        epoch_correct = 0
        num_instances = 0

        # set the model as training mode
        model.train()

        for instances in iterator:

            label_ids = torch.tensor([label_to_idx[instance['adjudicated_label']] for instance in instances]).to(device)

            optimizer.zero_grad()

            # make prediction
            preds = model(instances)
            preds = preds.squeeze(1)

            loss = criterion(preds, label_ids)
            epoch_loss += loss.item()

            # back-propogate
            loss.backward()
            optimizer.step()

            # calculate the accuracy
            pred_ids = preds.argmax(dim=1)
            epoch_correct += pred_ids.eq(label_ids).sum().item()
            num_instances += len(instances)

        return epoch_loss / num_instances, epoch_correct / num_instances

    def evaluate(self, model, iterator, criterion, device, label_to_idx):
        """Evaluate the trained model.

        Args:
            model: the trained model
            iterator: development data loader
            criterion: loss function
            device: cuda or cpu
            label_to_idx: convert the label into index to use the loss function
        Returns:
            the loss and accuracy on development data
        """
        epoch_loss = 0
        epoch_correct = 0
        num_instances = 0

        # set the model as evaluation mode
        model.eval()

        with torch.no_grad():

            for instances in iterator:
                label_ids = torch.tensor([label_to_idx[instance['adjudicated_label']] for instance in instances]).to(device)

                # use the model to predict
                preds = model(instances)
                preds = preds.squeeze(1)

                loss = criterion(preds, label_ids)
                epoch_loss += loss.item()

                # calculate the accuracy
                pred_ids = preds.argmax(dim=1)
                epoch_correct += pred_ids.eq(label_ids).sum().item()
                num_instances += len(instances)

        return epoch_loss / num_instances, epoch_correct / num_instances

