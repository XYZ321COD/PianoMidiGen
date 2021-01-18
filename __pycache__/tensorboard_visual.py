from tensorboard import program
from tensorboard import default
tb = program.TensorBoard(default.PLUGIN_LOADERS,
                         default.get_assets_zip_provider())
tb.configure(argv=['--logdir', "./logs"])
tb.main()
