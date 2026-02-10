using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Samples.Azure.Database.NL2SQL.Services;

public interface IKernelFactory
{
    (Kernel Kernel, IChatCompletionService ChatCompletion) CreateKernelForSession();
}
