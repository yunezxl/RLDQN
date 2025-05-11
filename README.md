Newsvendor Problems with Product Unbundling: An Approach Combining Robust Optimization with Deep Reinforcement Learning

Link to paper: https://doi.org/10.1177/10591478251344225

Abstract:

In fashion, food processing, petrochemical production, and agriculture, products (items) are often bundled in a pre-fixed assortment, with a given ratio for each product. For example, one case of men’s shoes may contain 24 pairs of different sizes of the same design. Of the 24 pairs, there is one size 7 pair, four sizes 9, and so on. Moreover, those pairs of shoes are packaged independently for retailing. Retailers of such products order them in bundles and then resell them unbundled. In this study, we propose and analyze a newsvendor model in which a retailer decides the order quantity of the whole bundle before the uncertain demand for each product/item is realized. We call it a product unbundling newsvendor problem (PUNP): How should the retailer decide the ordering quantity of a product bundle to meet the unknown demands of individual items to maximize its expected profit? We approach this problem with a robust optimization approach that assumes the means and covariance matrix of stochastic demands but not the demand distributions. However, the robust approach that considers the worst-case demand scenario is perceived to be conservative. In this study, we incorporate the distributionally robust optimization (DRO) with deep reinforcement learning (DRL) and propose a new paradigm of robust learning to improve the robust decision quality. We take this robust solution, i.e., the order quantity and profit, as human domain knowledge and implement it into the decision-making process of DRL by designing a policy transfer mechanism. Unsurprisingly, the exact robust solution is computationally intractable; thus, we provide an approximate solution. Simulations were conducted based on limited data sizes, confirming that our approach effectively improves robust performance. Moreover, the hybrid approach significantly outperforms the DRL approach. In the meantime, reduced computing costs and increased interpretability of decision recommendations may facilitate the deployment of DRL algorithms in operational practice. Furthermore, the successful application of the hybrid approach in addressing several variants of the PUNP indicates that the proposed mechanism may provide a pathway for solving complex operational problems.

Code:

TRAINER.py -- trains object from class unshaped_dqn, shaped_b or shaped_ble

UNSHAPED_DQN.py -- deep Q-network without reward shaping

SHAPED_B.py -- deep Q-network with reward shaping with base-stock policy as teacher

SHAPED_BLE.py -- deep Q-network with reward shaping with BSP-low-EW as teacher

ENV_TRAIN.py -- perishable inventory problem to train the DRL models

ENV_TEST.py -- perishable inventory problem to test the DRL models with seeded demand

EVALUATE_DRL_POLICY.py -- evaluates a trained DRL model in the test environment
