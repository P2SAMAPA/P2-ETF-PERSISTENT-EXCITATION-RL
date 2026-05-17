import numpy as np
from pe_actor_critic import PE_ActorCritic
from persistent_excitation import PersistentExcitation
from lyapunov import LyapunovMonitor
from config import PE_ADAPTIVE, BATCH_SIZE

def online_training(env, actor_critic, pe, lyap, epochs=10, batch_size=32):
    """Run online training loop with PE and Lyapunov checks."""
    for epoch in range(epochs):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Get adaptive noise from PE
            noise_std = pe.required_noise_std() if PE_ADAPTIVE else 0.01
            action = actor_critic.act(state, noise_std=noise_std)
            next_state, reward, done, _ = env.step(action)
            
            # Update PE with feature vector (state+action)
            feat = np.concatenate([state, action])
            pe.update(feat)
            
            actor_critic.remember(state, action, reward, next_state, done)
            actor_critic.train(batch_size)
            
            # Lyapunov check on actor parameters
            params = np.concatenate([p.detach().numpy().flatten() for p in actor_critic.actor.parameters()])
            stable, delta = lyap.update(params)
            if not stable:
                print(f"Lyapunov violation at step {epoch}, delta={delta:.4f}")
            state = next_state
            total_reward += reward
        print(f"Epoch {epoch}: total reward {total_reward:.2f}")
