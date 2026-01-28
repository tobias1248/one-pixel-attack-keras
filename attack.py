#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
from keras.datasets import cifar10
import pickle
import time

# Custom Networks
from networks.vit import ViT

# Helper functions
from differential_evolution import differential_evolution
import helper


class PixelAttacker:
    def __init__(self, models, data, class_names, dimensions=(32, 32)):
        # Load data and model
        self.models = models
        self.x_test, self.y_test = data
        self.class_names = class_names
        self.dimensions = dimensions

        network_stats, correct_imgs = helper.evaluate_models(self.models, self.x_test, self.y_test)
        self.correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
        self.network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

    def predict_classes(self, xs, img, target_class, model, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = helper.perturb_image(xs, img)
        predictions = model.predict(imgs_perturbed)[:, target_class]
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, target_class, model, targeted_attack=False, verbose=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = helper.perturb_image(x, img)

        confidence = model.predict(attack_image)[0]
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or 
        # targeted classification), return True
        if verbose:
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, img_id, model, target=None, pixel_count=1,
               maxiter=75, popsize=400, verbose=False, plot=False):
        start_time = time.perf_counter()
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img_id, 0]

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = self.dimensions
        bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            return self.predict_classes(xs, self.x_test[img_id], target_class, model, target is None)

        def callback_fn(x, convergence):
            return self.attack_success(x, self.x_test[img_id], target_class, model, targeted_attack, verbose)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        # Calculate some useful statistics to return from this function
        attack_image = helper.perturb_image(attack_result.x, self.x_test[img_id])[0]
        prior_probs = model.predict(np.array([self.x_test[img_id]]))[0]
        predicted_probs = model.predict(np.array([attack_image]))[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = self.y_test[img_id, 0]
        success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        # Show the best attempt at a solution (successful or not)
        if plot:
            helper.plot_image(attack_image, actual_class, self.class_names, predicted_class)

        duration = time.perf_counter() - start_time
        return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
                predicted_probs, attack_result.x, duration]

    def attack_all(self, models, samples=500, pixels=(1, 3, 5), targeted=False,
                   maxiter=75, popsize=400, verbose=False, sample_mode='random',
                   skip_correct_filter=False):
        results = []
        for model in models:
            model_results = []
            if skip_correct_filter:
                valid_imgs = np.arange(len(self.x_test))
            else:
                valid_imgs = self.correct_imgs[self.correct_imgs.name == model.name].img
            if sample_mode == 'first':
                img_samples = np.array(sorted(valid_imgs))[:samples]
            else:
                img_samples = np.random.choice(valid_imgs, samples)

            for pixel_count in pixels:
                for i, img in enumerate(img_samples):
                    print(model.name, '- image', img, '-', i + 1, '/', len(img_samples))
                    targets = [None] if not targeted else range(10)

                    for target in targets:
                        if targeted:
                            print('Attacking with target', self.class_names[target])
                            if target == self.y_test[img, 0]:
                                continue
                        result = self.attack(img, model, target, pixel_count,
                                             maxiter=maxiter, popsize=popsize,
                                             verbose=verbose)
                        model_results.append(result)

            results += model_results
            helper.checkpoint(results, targeted)
        return results


if __name__ == '__main__':
    def _load_lenet():
        from networks.lenet import LeNet
        return LeNet

    def _load_pure_cnn():
        from networks.pure_cnn import PureCnn
        return PureCnn

    def _load_net_in_net():
        from networks.network_in_network import NetworkInNetwork
        return NetworkInNetwork

    def _load_resnet():
        from networks.resnet import ResNet
        return ResNet

    def _load_densenet():
        from networks.densenet import DenseNet
        return DenseNet

    def _load_wide_resnet():
        from networks.wide_resnet import WideResNet
        return WideResNet

    def _load_capsnet():
        from networks.capsnet import CapsNet
        return CapsNet

    def _load_vit():
        return ViT

    model_defs = {
        'lenet': _load_lenet,
        'pure_cnn': _load_pure_cnn,
        'net_in_net': _load_net_in_net,
        'resnet': _load_resnet,
        'densenet': _load_densenet,
        'wide_resnet': _load_wide_resnet,
        'capsnet': _load_capsnet,
        'vit': _load_vit,
    }

    parser = argparse.ArgumentParser(description='Attack models on Cifar10')
    parser.add_argument('--model', nargs='+', choices=model_defs.keys(), default=model_defs.keys(),
                        help='Specify one or more models by name to evaluate.')
    parser.add_argument('--pixels', nargs='+', default=(1, 3, 5), type=int,
                        help='The number of pixels that can be perturbed.')
    parser.add_argument('--maxiter', default=75, type=int,
                        help='The maximum number of iterations in the differential evolution algorithm before giving up and failing the attack.')
    parser.add_argument('--popsize', default=400, type=int,
                        help='The number of adversarial images generated each iteration in the differential evolution algorithm. Increasing this number requires more computation.')
    parser.add_argument('--samples', default=500, type=int,
                        help='The number of image samples to attack. Images are sampled randomly from the dataset.')
    parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
    parser.add_argument('--sample_mode', choices=('random', 'first'), default='random',
                        help='Select image sampling mode. "first" uses the lowest-index correctly classified images.')
    parser.add_argument('--skip_correct_filter', action='store_true',
                        help='If set, sample from all test images without filtering to only correctly classified ones.')
    parser.add_argument('--save', default='networks/results/results.pkl', help='Save location for the results (pickle)')
    parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

    args = parser.parse_args()

    # Load data and model
    _, test = cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    models = [model_defs[m]()(load_weights=True) for m in args.model]

    attacker = PixelAttacker(models, test, class_names)

    print('Starting attack')

    results = attacker.attack_all(models, samples=args.samples, pixels=args.pixels, targeted=args.targeted,
                                  maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose,
                                  sample_mode=args.sample_mode,
                                  skip_correct_filter=args.skip_correct_filter)

    columns = ['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs',
               'perturbation', 'duration']
    results_table = pd.DataFrame(results, columns=columns)

    print(results_table[['model', 'pixels', 'image', 'true', 'predicted', 'success']])

    print('Saving to', args.save)
    with open(args.save, 'wb') as file:
        pickle.dump(results, file)
