using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using MQTTnet;
using MQTTnet.Client;
using Npgsql;
using IoTPlatform.Api.Models;

namespace IoTPlatform.Api.Services;

public class MqttIngestService : BackgroundService
{
    private readonly ILogger<MqttIngestService> _logger;
    private IMqttClient? _client;

    public MqttIngestService(ILogger<MqttIngestService> logger)
    {
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var factory = new MqttFactory();
        _client = factory.CreateMqttClient();

        var options = new MqttClientOptionsBuilder()
            .WithTcpServer("localhost", 1883)
            .Build();

        _client.ApplicationMessageReceivedAsync += async e =>
        {
            var payload = e.ApplicationMessage.Payload == null
                ? ""
                : Encoding.UTF8.GetString(e.ApplicationMessage.Payload);

            _logger.LogInformation("MQTT message received: {Payload}", payload);

            var record = JsonSerializer.Deserialize<TelemetryRecord>(
                payload,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true }
            );

            if (record == null)
            {
                _logger.LogError("JSON deserialize failed");
                return;
            }
            _logger.LogError($"DEBUG parent_device = '{record.Parent_Device}'");

            try
            {
                var connString = "Host=localhost;Port=5432;Database=iotdb;Username=iotuser;Password=iotpass";
                await using var conn = new NpgsqlConnection(connString);
                await conn.OpenAsync(stoppingToken);

                await using var cmd = new NpgsqlCommand(
                    @"INSERT INTO telemetry
                      (time, device_id, parent_device, temperature, humidity, power, device_state)
                      VALUES (@time, @device_id, @parent_device, @temperature, @humidity, @power, @device_state)",
                    conn
                );

                cmd.Parameters.AddWithValue("time", record.Time);
                cmd.Parameters.AddWithValue("device_id", record.Device_Id);
                cmd.Parameters.AddWithValue("parent_device", record.Parent_Device);
                cmd.Parameters.AddWithValue("temperature", record.Temperature);
                cmd.Parameters.AddWithValue("humidity", record.Humidity);
                cmd.Parameters.AddWithValue("power", record.Power);
                cmd.Parameters.AddWithValue("device_state", record.Device_State);

                await cmd.ExecuteNonQueryAsync(stoppingToken);
                _logger.LogInformation("Telemetry inserted into DB.");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "DB insert FAILED");
            }
        };

        await _client.ConnectAsync(options, stoppingToken);
        _logger.LogInformation("Connected to MQTT broker.");

        await _client.SubscribeAsync("iot/telemetry/#", cancellationToken: stoppingToken);
        _logger.LogInformation("Subscribed to iot/telemetry/#");

        while (!stoppingToken.IsCancellationRequested)
        {
            await Task.Delay(1000, stoppingToken);
        }
    }
}

