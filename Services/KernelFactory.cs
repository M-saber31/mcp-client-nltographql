using Azure.Identity;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

#pragma warning disable SKEXP0010

namespace Samples.Azure.Database.NL2SQL.Services;

public class KernelFactory : IKernelFactory
{
    private readonly McpClient _mcpClient;
    private readonly ILoggerFactory _loggerFactory;
    private readonly string _endpoint;
    private readonly string _apiKey;
    private readonly string _deploymentName;

    public KernelFactory(McpClient mcpClient, ILoggerFactory loggerFactory, string endpoint, string apiKey, string deploymentName)
    {
        _mcpClient = mcpClient;
        _loggerFactory = loggerFactory;
        _endpoint = endpoint;
        _apiKey = apiKey;
        _deploymentName = deploymentName;
    }

    public (Kernel Kernel, IChatCompletionService ChatCompletion) CreateKernelForSession()
    {
        var sc = new ServiceCollection();

        if (string.IsNullOrEmpty(_apiKey))
        {
            sc.AddAzureOpenAIChatCompletion(_deploymentName, _endpoint, new DefaultAzureCredential());
        }
        else
        {
            sc.AddAzureOpenAIChatCompletion(_deploymentName, _endpoint, _apiKey);
        }

        sc.AddKernel();
        sc.AddSingleton(_loggerFactory);
        sc.AddLogging();

        var services = sc.BuildServiceProvider();
        var kernel = services.GetRequiredService<Kernel>();
        var agentLogger = _loggerFactory.CreateLogger("GraphQLAgent");
        kernel.Plugins.AddFromObject(new GraphQLAgent(_mcpClient, agentLogger));

        var ai = kernel.GetRequiredService<IChatCompletionService>();
        return (kernel, ai);
    }
}
