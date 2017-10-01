#include <nn.h>

int main(int argc, char *argv[])
{
	struct nn_ffnet *net = nn_ffnet_create(2, 1, 2, 1);

	nn_ffnet_randomize(net);

	return 0;
}
