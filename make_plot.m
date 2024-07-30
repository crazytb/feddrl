clearvars

drqn = readmatrix('drqn_test_log.csv');
dqn = readmatrix('dqn_test_log.csv');
local = readmatrix('local_only_test_log.csv');
offload = readmatrix('offload_only_test_log.csv');

reward_drqn = drqn(20:20:end, 5);
reward_dqn = dqn(20:20:end, 5);
reward_local = local(20:20:end, 5);
reward_offload = offload(20:20:end, 5);

%%

x = 1:1:200;
edges = linspace(0, 500, 50);

figure()
hold on;
histogram(smoothdata(reward_drqn), edges, 'FaceColor', "#0072BD", 'LineWidth', 1)
histogram(smoothdata(reward_dqn), edges, 'FaceColor', "#D95319", 'LineWidth', 1)
histogram(smoothdata(reward_local), edges, 'FaceColor', "#EDB120", 'LineWidth', 1)
histogram(smoothdata(reward_offload), edges, 'FaceColor', "#7E2F8E", 'LineWidth', 1)
xline(median(reward_drqn), 'Color', "#0072BD", 'LineWidth', 2);
xline(median(reward_dqn), 'Color', "#D95319", 'LineWidth', 2);
xline(median(reward_local), 'Color', "#EDB120", 'LineWidth', 2);
xline(median(reward_offload), 'Color', "#7E2F8E", 'LineWidth', 2);

legend('PORTO-MEC (DRQN)', 'OTO-MEC (DQN)', 'Local-only', 'Offload-only', 'Location', 'best')
xlim([0 500])
xlabel('Episode Rewards')
ylabel('Frequency')
title('')
grid on
hold off;

disp(mean(reward_drqn))
disp(mean(reward_dqn))
disp(mean(reward_local))
disp(mean(reward_offload))

disp(std(reward_drqn))
disp(std(reward_dqn))
disp(std(reward_local))
disp(std(reward_offload))

%%

drqn_train = readmatrix('DRQN_POMDP_Random_20240730_173601.csv');
dqn_train = readmatrix('DQN_POMDP_20240730_174713.csv');
drqn_train = drqn_train(:, 2:3);
dqn_train = dqn_train(:, 2:3);
window_size = 200;

figure()
hold on;
plot(drqn_train(:, 1), movmean(drqn_train(:, 2), window_size), 'Color', "#0072BD", 'LineWidth', 2)
plot(dqn_train(:, 1), movmean(dqn_train(:, 2), window_size), 'Color', "#D95319", 'LineWidth', 2)
legend('PORTO-MEC (DRQN)', 'OTO-MEC (DQN)', 'location', 'best')
xlabel('Episode Rewards')
ylabel('Frequency')
title('')
grid on
hold off;