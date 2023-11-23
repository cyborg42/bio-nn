use crossbeam::channel;
use dashmap::DashMap;

use std::sync::Arc;
use std::thread::sleep;
use std::time::{self, Duration};
use std::{mem, thread};

#[derive(Debug)]
struct Neuron {
    id: usize,
    energy: f64,
    activity: f64,
    last_time: time::Instant,
    sleep_time_ms: u64,
    sends: Vec<(usize, f64)>,
    recv: channel::Receiver<f64>,
    shared_info: Arc<SharedInfo>,
}

impl Neuron {
    fn start(&mut self) {
        let rand = fastrand::f64();
        let now = time::Instant::now();
        let duration = now.duration_since(self.last_time).as_nanos() as f64 / 1_000_000_000.;
        self.last_time = now;
        let mut recv = self.recv.try_iter().reduce(|a, b| a + b).unwrap_or(0.);
        recv += (rand * 2. - 1.) * duration;

        self.activity *= f64::powf(0.5, duration);
        self.energy = (self.energy + duration).min(self.shared_info.max_energy);
        if rand < (self.activity + 0.1) * duration {
            let id = fastrand::usize(0..self.shared_info.size);
            if self.sends.len() < self.shared_info.max_link
                && id != self.id
                && rand < *self.shared_info.neurons_activity.get(&id).unwrap() + 0.1
            {
                self.sends.push((id, rand * 2. - 1.));
            }
        }

        if recv > self.shared_info.threshold * duration {
            self.activity += recv.abs();
            //self.activity = self.activity.min(1.);
            self.sleep_time_ms = 1;
            for (id, weight) in self.sends.iter_mut() {
                self.shared_info.sends[*id]
                    .send(recv * *weight * self.energy)
                    .unwrap();
                *weight += *weight / weight.abs()
                    * duration
                    * self.activity
                    * *self.shared_info.neurons_activity.get(id).unwrap();
                //*weight = (*weight).min(1.);
            }
            self.energy = 0.;
        } else {
            self.sends = self
                .sends
                .iter()
                .filter_map(|(id, weight)| {
                    let x = (*id, weight * f64::powf(0.5, duration));
                    if x.1.abs() > 0.01 {
                        Some(x)
                    } else {
                        None
                    }
                })
                .collect();

            self.sleep_time_ms = (self.sleep_time_ms * 2).min(1000);
        }
        self.shared_info
            .neurons_activity
            .insert(self.id, self.activity);
        sleep(Duration::from_millis(self.sleep_time_ms));
        if self.id == 1{
            dbg!(
                self.id,
                self.activity,
                self.energy,
                self.sends.clone(),
                recv
            );
        }
    }
}

struct NeuralNetwork {
    neurons: Vec<Neuron>,
    shared_info: Arc<SharedInfo>,
}
#[derive(Debug)]
struct SharedInfo {
    max_energy: f64,
    threshold: f64,
    max_link: usize,
    size: usize,
    sends: Vec<channel::Sender<f64>>,
    neurons_activity: DashMap<usize, f64>,
}

impl NeuralNetwork {
    fn new(size: usize, max_energy: f64, threshold: f64, max_link: usize) -> Self {
        let mut shared_info = SharedInfo {
            max_energy,
            threshold,
            max_link,
            size,
            sends: Vec::with_capacity(size),
            neurons_activity: DashMap::new(),
        };
        let mut recvs = Vec::with_capacity(size);
        for id in 0..size {
            let (send, recv) = channel::unbounded();
            shared_info.sends.push(send);
            recvs.push(recv);
            shared_info.neurons_activity.insert(id, 0.);
        }
        let shared_info = Arc::new(shared_info);
        let neurons = recvs
            .into_iter()
            .enumerate()
            .map(|(id, recv)| Neuron {
                id,
                energy: fastrand::f64(),
                activity: fastrand::f64(),
                last_time: time::Instant::now(),
                sleep_time_ms: 1,
                sends: Vec::new(),
                recv,
                shared_info: shared_info.clone(),
            })
            .collect();
        let mut neurons_activity = Vec::with_capacity(size);
        neurons_activity.resize(size, 0.);
        Self {
            neurons,
            shared_info,
        }
    }
    fn start(&mut self) {
        let neurons = mem::take(&mut self.neurons);
        for mut neuron in neurons.into_iter() {
            thread::spawn(move || loop {
                neuron.start();
            });
        }
    }
    fn report(&self) {
        for activity in self.shared_info.neurons_activity.iter() {
            print!("{}: {:.4}\t", activity.key(), activity.value());
        }
        println!();
    }
}

fn main() {
    let mut nn = NeuralNetwork::new(100, 10., 0.5, 20);
    nn.start();
    loop {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        nn.report();
    }
}
