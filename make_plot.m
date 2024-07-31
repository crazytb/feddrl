clearvars

drqn = readcell('test_log_drqn.csv');
dqn = readcell('test_log_dqn.csv');
local = readcell('test_log_local_only.csv');
offload = readcell('test_log_offload_only.csv');

max_epoch = 200;

reward_drqn = cell2mat(drqn(max_epoch:max_epoch:end, 5));
reward_dqn = cell2mat(dqn(max_epoch:max_epoch:end, 5));
reward_local = cell2mat(local(max_epoch:max_epoch:end, 5));
reward_offload = cell2mat(offload(max_epoch:max_epoch:end, 5));

%%

edges = linspace(0, 1500, 50);

figure()
hold on;
histogram(smoothdata(reward_drqn), edges, 'FaceColor', "#0072BD", 'LineWidth', 1)
histogram(smoothdata(reward_dqn), edges, 'FaceColor', "#D95319", 'LineWidth', 1)
histogram(smoothdata(reward_offload), edges, 'FaceColor', "#7E2F8E", 'LineWidth', 1)
histogram(smoothdata(reward_local), edges, 'FaceColor', "#EDB120", 'LineWidth', 1)
xline(median(reward_drqn), 'Color', "#0072BD", 'LineWidth', 4);
xline(median(reward_dqn), 'Color', "#D95319", 'LineWidth', 4);
xline(median(reward_offload), 'Color', "#7E2F8E", 'LineWidth', 4);
xline(median(reward_local), 'Color', "#EDB120", 'LineWidth', 4);

legend('PORTO-MEC (DRQN)', 'OTO-MEC (DQN)', 'Offload-only', 'Local-only', 'Location', 'best')
xlim([0 1500])
xlabel('Episode Rewards')
ylabel('Frequency')
title('')
grid on
hold off;

disp(mean(reward_drqn))
disp(mean(reward_dqn))
disp(mean(reward_offload))
disp(mean(reward_local))

disp(std(reward_drqn))
disp(std(reward_dqn))
disp(std(reward_offload))
disp(std(reward_local))

%%

drqn_train = readmatrix('DRQN_POMDP_Random_20240731_162618.csv');
dqn_train = readmatrix('DQN_POMDP_20240731_163431.csv');
drqn_train = drqn_train(:, 2:3);
dqn_train = dqn_train(:, 2:3);
window_size = 500;

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