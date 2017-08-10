extern crate rand;

use rand::Rng;

#[allow(dead_code)]
enum ActivationModes {
	IDENTITY,
	BINARY,
	LOGISTIC,
	TANH,
	RECTIFIED,
	GAUSSIAN
}

struct Dendrite { /* Dendrite: The input layer of a single neuron */
	d: f64, /* Data         */
	w: f64, /* Weight       */
	dw: f64 /* Delta weight */
}

struct Synapse { /* Synapse: The output layer of a single neuron */
	raw: f64,                             /* Net value from inputs                                     */
	output: f64,                          /* Final output after activation function                    */
	error: f64,                           /* Error value (depends on desired value)                    */
	desired: f64,                         /* What the output should be                                 */
	error_precision: f64,                 /* How low can the error go until it is considered valid     */
	threshold: f64,                       /* Threshold value used by the activation function           */
	activation_function: ActivationModes, /* What activation function should be used for the raw value */
}

struct Neuron { /* The neuron */
	inputs: Vec<Dendrite>, /* Array of dendrites (inputs of the neuron)        */
	bias: f64,             /* Bias value to add to the dendrites's input value */
	learning_coeff: f64,   /* The learning rate of the neuron                  */
	synapse: Synapse,      /* Output layer of the neuron                       */
	good: bool,            /* Is the synapse output good enough                */
	propagation_count: u64 /* How many times it propagated                     */
}

fn propagate(neuron: &mut Neuron, train: bool) {
	/* Reset internal net (raw) value from last propagation calculation */
	neuron.synapse.raw = 0.0;

	/* Calculate new synapse raw value */
	for dendrite in &mut neuron.inputs {
		neuron.synapse.raw += dendrite.d * dendrite.w; /* net = net + (Data * Weight) */
	}

	/* Add bias */
	neuron.synapse.raw += neuron.bias; /* net = net + bias */

	/* Calculate final output using a specific activation function */
	neuron.synapse.output = match neuron.synapse.activation_function {
		ActivationModes::IDENTITY  => {
			neuron.synapse.raw
		},
		ActivationModes::BINARY    => {
			if neuron.synapse.raw < neuron.synapse.threshold
				{ 0.0 }
			else 
				{ 1.0 }
		},
		ActivationModes::LOGISTIC  => {
			1.0 / (1.0 + (-1.0 * neuron.synapse.raw).exp())
		}
		ActivationModes::TANH      => {
			neuron.synapse.raw.tanh()
		},
		ActivationModes::RECTIFIED => {
			if neuron.synapse.raw < neuron.synapse.threshold
				{ 0.0 }
			else 
				{ neuron.synapse.raw }
		},
		ActivationModes::GAUSSIAN  => {
			(-1.0 * neuron.synapse.raw.powi(2)).exp()
		}
	};

	/* Calculate error value */
	neuron.synapse.error = neuron.synapse.desired - neuron.synapse.output;

	/* Calculate delta weight for the next training cycle */
	for dendrite in &mut neuron.inputs {
		dendrite.dw = dendrite.d * neuron.learning_coeff * neuron.synapse.error;
		if train {
			dendrite.w += dendrite.dw;
		}
	}

	/* Update valid flag */
	neuron.good = 
		if neuron.synapse.error <= neuron.synapse.error_precision.abs() && neuron.synapse.error >= -neuron.synapse.error_precision.abs()
			{ true }
		else 
			{ false };

	/* And we're done */
	neuron.propagation_count += 1;
}

#[allow(dead_code)]
fn propagate_only(neuron: &mut Neuron) {
	propagate(neuron, false);
}

#[allow(dead_code)]
fn propagate_and_train(neuron: &mut Neuron) {
	propagate(neuron, true);
}

fn create_neuron(
	input_data: Vec<f64>, bias: f64,
	desired: f64, error_precision: f64, threshold: f64,
	learning_coeff: f64, activation_function: ActivationModes) -> Neuron
{
	/* Create dendrites from the given data */
	let mut dendrites = Vec::new();

	for input in &input_data {
		dendrites.push(Dendrite{d: *input, w: rand::thread_rng().gen_range(-1.0, 1.0), dw: 0.0});
	}

	Neuron {
		inputs: dendrites,              /* Initialise dendrites         */ 
		bias: bias,                     /* Set bias value               */
		learning_coeff: learning_coeff, /* Set the learning coefficient */
		synapse:                        /* Set Synapses                 */
		Synapse {
			raw:     0.0,
			output:  0.0,
			error:   0.0,
			desired: desired,
			error_precision: error_precision,
			threshold: threshold,
			activation_function: activation_function
		},
		good: false,         /* Reset valid flag        */
		propagation_count: 0 /* Reset propagation count */
	}
}

#[allow(non_upper_case_globals)]
static mut iteration: u32 = 0;
#[allow(non_upper_case_globals)]
static mut epoch: u32 = 0;
static MAX_TRAIN_COUNT: u32 = 10000;
static CONFIRM_TRAIN_COUNT: u32 = 5; 

unsafe fn dump_neuron(neuron: &mut Neuron) {
	println!("Iteration {} Epoch {}) Raw: {} Output: {} Desired: {} Error: {}, Valid: {}  (w0: {} w1: {})", 
		iteration, epoch, neuron.synapse.raw, neuron.synapse.output, neuron.synapse.desired, neuron.synapse.error, neuron.good, neuron.inputs[1].w, neuron.inputs[0].w);
}

unsafe fn cycle_and_dump(neuron: &mut Neuron) {
	propagate(neuron, true);
	dump_neuron(neuron);
	iteration += 1;
}

unsafe fn train_until_done(neuron: &mut Neuron, max_epochs: u32) -> u32 {
	let mut valid_counter = 0;
	let mut counter = 0;
	#[allow(unused_variables)]
	for i in 0..max_epochs {
		counter = i;
		cycle_and_dump(neuron);
		if neuron.good {
			valid_counter += 1;
			if valid_counter >= CONFIRM_TRAIN_COUNT {
				println!("");
				return counter + 1;
			}
		}
	}
	println!("");
	counter
}

fn update_neuron_data(neuron: &mut Neuron, dataset: &Vec<f64>, desired: f64) {
	for i in 0..neuron.inputs.len() {
		neuron.inputs[i].d = dataset[i];
	}
	neuron.synapse.desired = desired;
}

fn main() {
	unsafe {
		/************************/
		/* Create single neuron */
		/************************/
		let mut neuron1 = 
			create_neuron(
				vec![ /* Input data */
					0.5, 0.5, 0.5, 0.0
				],
				0.0,    /* Bias                 */
				10.0,   /* Desired output value */
				0.0001, /* Error precision      */
				0.0,    /* Threshold value      */
				0.5,    /* Learning coefficient */
				ActivationModes::RECTIFIED /* Activation function */
			);

		/****************/
		/* Train neuron */
		/****************/
		train_until_done(&mut neuron1, MAX_TRAIN_COUNT);
		println!("Done training\n\n");

		/***************************************/
		/* Train neuron for other sets of data */
		/***************************************/
		let mut all_trained: u32 = 0;

		let dataset: Vec<(Vec<f64>, f64)> = vec![
			(vec![0.0,0.4,0.5,0.0], 20.0),
			(vec![0.2,0.7,0.0,0.0], 30.0),
			(vec![0.8,0.1,0.5,1.0], 10.0),
		];

		#[allow(unused_variables)]
		for i in 0..10000 {
			for j in 0..dataset.len() {
				update_neuron_data(&mut neuron1, &dataset[j].0, dataset[j].1);
				if train_until_done(&mut neuron1, MAX_TRAIN_COUNT) == CONFIRM_TRAIN_COUNT {
					all_trained += 1;
				}
			}

			epoch += 1;
			iteration = 0;
			
			if all_trained == dataset.len() as u32 {
				break;
			}
			all_trained = 0;
		}

		println!("Done training all datasets\n");
	}
}
